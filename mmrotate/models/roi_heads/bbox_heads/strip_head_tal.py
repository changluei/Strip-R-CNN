# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy

from ...builder import ROTATED_HEADS
from .strip_head import StripHead


@ROTATED_HEADS.register_module()
class StripHeadTAL(StripHead):
    """StripHead with TAL-aware target weighting for second-stage training."""

    def get_targets_tal(self,
                        sampling_results,
                        gt_bboxes,
                        gt_labels,
                        rcnn_train_cfg,
                        concat=True):
        del gt_bboxes, gt_labels, rcnn_train_cfg
        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []

        for res in sampling_results:
            pos_bboxes = res.pos_bboxes
            neg_bboxes = res.neg_bboxes
            pos_gt_bboxes = res.pos_gt_bboxes
            pos_gt_labels = res.pos_gt_labels

            num_pos = pos_bboxes.size(0)
            num_neg = neg_bboxes.size(0)
            num_samples = num_pos + num_neg
            box_dim = pos_bboxes.size(1)

            labels = pos_bboxes.new_full((num_samples, ),
                                         self.num_classes,
                                         dtype=torch.long)
            label_weights = pos_bboxes.new_zeros(num_samples)
            bbox_targets = pos_bboxes.new_zeros((num_samples, box_dim))
            bbox_weights = pos_bboxes.new_zeros((num_samples, box_dim))

            if num_pos > 0:
                labels[:num_pos] = pos_gt_labels
                if self.reg_decoded_bbox:
                    pos_bbox_targets = pos_gt_bboxes
                else:
                    pos_bbox_targets = self.bbox_coder.encode(
                        pos_bboxes, pos_gt_bboxes)
                bbox_targets[:num_pos, :] = pos_bbox_targets

                pos_t_hat = getattr(res, 'pos_t_hat', pos_bboxes.new_ones(num_pos))
                pos_t_hat = pos_t_hat.detach().clamp(min=1e-3, max=1.0)
                label_weights[:num_pos] = pos_t_hat
                bbox_weights[:num_pos, :] = pos_t_hat.unsqueeze(1)

            if num_neg > 0:
                label_weights[num_pos:] = 1.0

            labels_list.append(labels)
            label_weights_list.append(label_weights)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)

        if concat:
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)
            return labels, label_weights, bbox_targets, bbox_weights

        return labels_list, label_weights_list, bbox_targets_list, bbox_weights_list

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        if cls_score is not None:
            cls_avg_factor = max(label_weights.sum().item(), 1.0)
            if cls_score.numel() > 0:
                loss_cls = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls, dict):
                    losses.update(loss_cls)
                else:
                    losses['loss_cls'] = loss_cls
                if self.custom_activation:
                    losses.update(self.loss_cls.get_accuracy(cls_score, labels))
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            pos_inds = pos_inds.type(torch.bool)

            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)

                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 5)[pos_inds, labels[pos_inds]]

                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_weights = bbox_weights[pos_inds]
                reg_avg_factor = max(pos_bbox_weights.sum().item(), 1.0)
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    pos_bbox_targets,
                    pos_bbox_weights,
                    avg_factor=reg_avg_factor,
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0

        return losses
