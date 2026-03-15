# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv import print_log
from mmcv.runner import get_dist_info

from mmrotate.core import rbbox2roi
from mmrotate.utils import get_root_logger

from ..builder import ROTATED_HEADS
from .oriented_standard_roi_head import OrientedStandardRoIHead


@ROTATED_HEADS.register_module()
class TaskAlignedOrientedStandardRoIHeadAssignOnly(OrientedStandardRoIHead):
    """TAL assignment-only oriented RoI head.

    This head only changes second-stage assign/sample by using TAL-style
    assignment based on second-stage pre-forward predictions, then falls back
    to the original bbox target/loss path.
    """

    def __init__(self,
                 append_gt_as_proposals=True,
                 debug_assign_stats=False,
                 debug_assign_interval=200,
                 debug_assign_log_first_n=5,
                 *args,
                 **kwargs):
        self.append_gt_as_proposals = append_gt_as_proposals
        self.debug_assign_stats = debug_assign_stats
        self.debug_assign_interval = max(int(debug_assign_interval), 1)
        self.debug_assign_log_first_n = max(int(debug_assign_log_first_n), 0)
        self._debug_assign_iter = 0
        super(TaskAlignedOrientedStandardRoIHeadAssignOnly,
              self).__init__(*args, **kwargs)

    @staticmethod
    def _get_assign_extra(assign_result, key, default=None):
        if hasattr(assign_result, 'get_extra_property'):
            value = assign_result.get_extra_property(key)
            return default if value is None else value
        return getattr(assign_result, key, default)

    @staticmethod
    def _fmt_triplet(stats):
        return f"{stats['mean']:.4f}/{stats['min']:.4f}/{stats['max']:.4f}"

    def _collect_assign_debug_item(self, assign_result, sampling_result, img_meta):
        debug = self._get_assign_extra(assign_result, 'assign_debug', {})
        filename = img_meta.get('ori_filename', img_meta.get('filename', 'unknown'))
        return dict(
            filename=filename,
            num_props=int(debug.get('num_props', 0)),
            num_gts=int(debug.get('num_gts', 0)),
            cls_nonfinite=int(debug.get('cls_nonfinite', 0)),
            iou_nonfinite=int(debug.get('iou_nonfinite', 0)),
            s_stats=debug.get('s_stats', dict(mean=0.0, min=0.0, max=0.0)),
            u_stats=debug.get('u_stats', dict(mean=0.0, min=0.0, max=0.0)),
            t_stats=debug.get('t_stats', dict(mean=0.0, min=0.0, max=0.0)),
            candidate_per_gt=debug.get(
                'candidate_per_gt', dict(mean=0.0, min=0, max=0)),
            multi_match_props=int(debug.get('multi_match_props', 0)),
            assigned_pos=int(debug.get('assigned_pos', 0)),
            assigned_neg=int(debug.get('assigned_neg', 0)),
            sampled_pos=int(sampling_result.pos_inds.numel()),
            sampled_neg=int(sampling_result.neg_inds.numel()),
            t_hat_pos_stats=debug.get(
                't_hat_pos_stats', dict(mean=0.0, min=0.0, max=0.0)))

    def _log_assign_debug(self, debug_items):
        if not debug_items:
            return
        num_imgs = len(debug_items)
        total_props = sum(item['num_props'] for item in debug_items)
        total_gts = sum(item['num_gts'] for item in debug_items)
        total_assigned_pos = sum(item['assigned_pos'] for item in debug_items)
        total_sampled_pos = sum(item['sampled_pos'] for item in debug_items)
        total_cls_nonfinite = sum(item['cls_nonfinite'] for item in debug_items)
        total_iou_nonfinite = sum(item['iou_nonfinite'] for item in debug_items)
        total_multi_match = sum(item['multi_match_props'] for item in debug_items)

        s_mean = sum(item['s_stats']['mean'] for item in debug_items) / num_imgs
        u_mean = sum(item['u_stats']['mean'] for item in debug_items) / num_imgs
        t_mean = sum(item['t_stats']['mean'] for item in debug_items) / num_imgs
        cand_mean = sum(item['candidate_per_gt']['mean']
                        for item in debug_items) / num_imgs

        worst_item = max(debug_items, key=lambda x: x['t_stats']['max'])
        message = (
            f"[AssignDebug][iter={self._debug_assign_iter}] "
            f"imgs={num_imgs} props={total_props} gts={total_gts} "
            f"assigned_pos={total_assigned_pos} sampled_pos={total_sampled_pos} "
            f"cls_nonfinite={total_cls_nonfinite} iou_nonfinite={total_iou_nonfinite} "
            f"multi_match={total_multi_match} "
            f"s(mean)={s_mean:.4f} u(mean)={u_mean:.4f} t(mean)={t_mean:.4f} "
            f"cand_per_gt(mean)={cand_mean:.2f} "
            f"worst_t_img={worst_item['filename']} "
            f"s(m/n/x)={self._fmt_triplet(worst_item['s_stats'])} "
            f"u(m/n/x)={self._fmt_triplet(worst_item['u_stats'])} "
            f"t(m/n/x)={self._fmt_triplet(worst_item['t_stats'])} "
            f"t_hat_pos(m/n/x)={self._fmt_triplet(worst_item['t_hat_pos_stats'])}")
        print_log(message, logger=get_root_logger())

    def _get_empty_bbox_outputs(self, proposals):
        if self.bbox_head.custom_cls_channels:
            cls_channels = self.bbox_head.loss_cls.get_cls_channels(
                self.bbox_head.num_classes)
        else:
            cls_channels = self.bbox_head.num_classes + 1
        reg_channels = 5 if self.bbox_head.reg_class_agnostic else \
            5 * self.bbox_head.num_classes
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

        self._debug_assign_iter += 1
        rank, _ = get_dist_info()
        log_assign_stats = (
            self.debug_assign_stats and rank == 0 and
            (self._debug_assign_iter <= self.debug_assign_log_first_n or
             self._debug_assign_iter % self.debug_assign_interval == 0))
        debug_items = [] if log_assign_stats else None

        sampling_results = []
        num_imgs = len(img_metas)

        for img_idx in range(num_imgs):
            proposals = proposal_list[img_idx][:, :5]
            gt_bboxes_i = gt_bboxes[img_idx][:, :5]
            gt_labels_i = gt_labels[img_idx]

            if self.append_gt_as_proposals and gt_bboxes_i.numel() > 0:
                all_proposals = torch.cat([proposals, gt_bboxes_i], dim=0)
            else:
                all_proposals = proposals

            # Pre-forward all proposals for TAL assignment metric.
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
                img_meta=img_metas[img_idx],
                gt_bboxes_ignore=gt_bboxes_ignore[img_idx])

            sampling_result = self.bbox_sampler.sample(
                assign_result,
                all_proposals,
                gt_bboxes_i,
                gt_labels_i,
                feats=[lvl_feat[img_idx][None] for lvl_feat in x])

            if gt_bboxes_i.numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes_i.new_zeros((0, 5))
            else:
                sampling_result.pos_gt_bboxes = gt_bboxes_i[
                    sampling_result.pos_assigned_gt_inds, :]
            sampling_results.append(sampling_result)
            if log_assign_stats:
                debug_items.append(
                    self._collect_assign_debug_item(assign_result,
                                                    sampling_result,
                                                    img_metas[img_idx]))

        # Use the original bbox training path (original targets + original loss).
        if log_assign_stats:
            self._log_assign_debug(debug_items)
        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes,
                                                gt_labels, img_metas)
        losses.update(bbox_results['loss_bbox'])
        return losses
