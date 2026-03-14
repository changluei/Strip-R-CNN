# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrotate.core import rbbox2roi

from ..builder import ROTATED_HEADS
from .oriented_standard_roi_head import OrientedStandardRoIHead


@ROTATED_HEADS.register_module()
class TaskAlignedOrientedStandardRoIHead(OrientedStandardRoIHead):
    """Oriented RoI head with TAL-style assignment for second-stage training."""

    def __init__(self, append_gt_as_proposals=True, *args, **kwargs):
        self.append_gt_as_proposals = append_gt_as_proposals
        super(TaskAlignedOrientedStandardRoIHead, self).__init__(*args, **kwargs)

    def _get_empty_bbox_outputs(self, proposals):
        if self.bbox_head.custom_cls_channels:
            cls_channels = self.bbox_head.loss_cls.get_cls_channels(
                self.bbox_head.num_classes)
        else:
            cls_channels = self.bbox_head.num_classes + 1

        reg_channels = 5 if self.bbox_head.reg_class_agnostic else 5 * self.bbox_head.num_classes
        cls_score = proposals.new_zeros((0, cls_channels))
        bbox_pred = proposals.new_zeros((0, reg_channels))
        return cls_score, bbox_pred

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        del gt_masks
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(len(img_metas))]

        losses = dict()
        if not self.with_bbox:
            return losses

        sampling_results = []
        num_imgs = len(img_metas)

        for img_idx in range(num_imgs):
            proposals = proposal_list[img_idx][:, :5]
            gt_bboxes_i = gt_bboxes[img_idx]
            gt_labels_i = gt_labels[img_idx]

            if self.append_gt_as_proposals and gt_bboxes_i.numel() > 0:
                all_proposals = torch.cat([proposals, gt_bboxes_i], dim=0)
            else:
                all_proposals = proposals

            x_single = [lvl_feat[img_idx:img_idx + 1] for lvl_feat in x]
            rois_all = rbbox2roi([all_proposals])
            if rois_all.numel() == 0:
                cls_score_all, bbox_pred_all = self._get_empty_bbox_outputs(
                    all_proposals)
            else:
                pre_bbox_results = self._bbox_forward(x_single, rois_all)
                cls_score_all = pre_bbox_results['cls_score']
                bbox_pred_all = pre_bbox_results['bbox_pred']

            assign_result = self.bbox_assigner.assign(
                proposals=all_proposals,
                cls_score=cls_score_all,
                bbox_pred=bbox_pred_all,
                gt_bboxes=gt_bboxes_i,
                gt_labels=gt_labels_i,
                bbox_coder=self.bbox_head.bbox_coder,
                img_meta=img_metas[img_idx])

            sampling_result = self.bbox_sampler.sample(
                assign_result,
                all_proposals,
                gt_bboxes_i,
                gt_labels_i,
                feats=[lvl_feat[img_idx][None] for lvl_feat in x])

            box_dim = all_proposals.size(1)
            if gt_bboxes_i.numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes_i.new_zeros((0, box_dim))
            else:
                sampling_result.pos_gt_bboxes = gt_bboxes_i[
                    sampling_result.pos_assigned_gt_inds, :]

            assign_t_hat = getattr(
                assign_result, 'assign_t_hat',
                all_proposals.new_zeros(all_proposals.size(0)))
            assign_t = getattr(
                assign_result, 'assign_metrics',
                all_proposals.new_zeros(all_proposals.size(0)))
            assign_u = getattr(
                assign_result, 'assign_ious',
                all_proposals.new_zeros(all_proposals.size(0)))
            sampling_result.pos_t_hat = assign_t_hat[sampling_result.pos_inds]
            sampling_result.pos_t = assign_t[sampling_result.pos_inds]
            sampling_result.pos_u = assign_u[sampling_result.pos_inds]

            sampling_results.append(sampling_result)

        bbox_results = self._bbox_forward_train_tal(x, sampling_results, gt_bboxes,
                                                    gt_labels, img_metas)
        losses.update(bbox_results['loss_bbox'])
        return losses

    def _bbox_forward_train_tal(self, x, sampling_results, gt_bboxes, gt_labels,
                                img_metas):
        del img_metas
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets_tal(
            sampling_results=sampling_results,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            rcnn_train_cfg=self.train_cfg)
        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
