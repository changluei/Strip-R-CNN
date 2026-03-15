# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator


def _set_assign_extra(assign_result, key, value):
    """Attach extra tensors to AssignResult in a version-safe way."""
    if hasattr(assign_result, 'set_extra_property'):
        assign_result.set_extra_property(key, value)
    else:
        setattr(assign_result, key, value)


@ROTATED_BBOX_ASSIGNERS.register_module()
class TaskAlignedRRoIAssigner(BaseAssigner):
    """RoI-level TAL assigner for oriented second-stage training."""

    def __init__(self,
                 topk=6,
                 alpha=1.0,
                 beta=6.0,
                 candidate_iou_thr=0.0,
                 use_max_t_when_conflict=True,
                 iou_calculator=dict(type='RBboxOverlaps2D'),
                 eps=1e-12,
                 **kwargs):
        # Keep compatibility with config inheritance where base assigner keys
        # (e.g., pos_iou_thr/neg_iou_thr) may be merged in.
        del kwargs
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.candidate_iou_thr = candidate_iou_thr
        self.use_max_t_when_conflict = use_max_t_when_conflict
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.eps = eps

    @staticmethod
    def _tensor_stats(x):
        if x.numel() == 0:
            return dict(mean=0.0, min=0.0, max=0.0)
        return dict(
            mean=float(x.mean().item()),
            min=float(x.min().item()),
            max=float(x.max().item()))

    def _empty_result(self, num_gts, num_props, device):
        assigned_gt_inds = torch.zeros(num_props, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_props, ), -1, dtype=torch.long, device=device)
        max_overlaps = torch.zeros(num_props, dtype=torch.float, device=device)
        assign_result = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)
        zeros = torch.zeros(num_props, dtype=torch.float, device=device)
        _set_assign_extra(assign_result, 'assign_metrics', zeros)
        _set_assign_extra(assign_result, 'assign_ious', zeros)
        _set_assign_extra(assign_result, 'assign_t_hat', zeros)
        _set_assign_extra(
            assign_result, 'assign_debug',
            dict(
                num_props=int(num_props),
                num_gts=int(num_gts),
                cls_nonfinite=0,
                iou_nonfinite=0,
                s_stats=dict(mean=0.0, min=0.0, max=0.0),
                u_stats=dict(mean=0.0, min=0.0, max=0.0),
                t_stats=dict(mean=0.0, min=0.0, max=0.0),
                candidate_per_gt=dict(mean=0.0, min=0, max=0),
                multi_match_props=0,
                assigned_pos=0,
                assigned_neg=int(num_props),
                t_hat_pos_stats=dict(mean=0.0, min=0.0, max=0.0)))
        return assign_result

    def _decode_iou_matrix(self, proposals, bbox_pred, gt_bboxes, gt_labels,
                           bbox_coder):
        """Decode predictions and return IoU matrix with shape [N, M]."""
        num_props = proposals.size(0)
        num_gts = gt_bboxes.size(0)

        # Class-agnostic regression: [N, 5]
        if bbox_pred.size(1) == 5:
            decoded_bboxes = bbox_coder.decode(proposals, bbox_pred)
            return self.iou_calculator(decoded_bboxes, gt_bboxes)

        # Class-specific regression: [N, C * 5]
        decoded = bbox_coder.decode(proposals, bbox_pred).view(num_props, -1, 5)
        cls_inds = gt_labels.clamp(min=0, max=decoded.size(1) - 1)
        decoded_for_gt = decoded[:, cls_inds, :]  # [N, M, 5]

        ious = proposals.new_zeros((num_props, num_gts), dtype=torch.float)
        for gt_idx in range(num_gts):
            # IoU between proposal predictions for gt class and one gt box.
            ious[:, gt_idx] = self.iou_calculator(
                decoded_for_gt[:, gt_idx, :], gt_bboxes[gt_idx:gt_idx + 1]).squeeze(1)
        return ious

    @torch.no_grad()
    def assign(self,
               proposals,
               cls_score,
               bbox_pred,
               gt_bboxes,
               gt_labels,
               bbox_coder,
               gt_bboxes_ignore=None,
               img_meta=None,
               **kwargs):
        """Assign gt to proposals with TAL metric."""
        del gt_bboxes_ignore, img_meta  # reserved for compatibility
        proposals = proposals[:, :5]
        gt_bboxes = gt_bboxes[:, :5]
        device = proposals.device
        num_props = proposals.size(0)
        num_gts = gt_bboxes.size(0)

        if num_props == 0 or num_gts == 0:
            return self._empty_result(num_gts, num_props, device)

        gt_labels = gt_labels.to(device=device, dtype=torch.long)
        cls_nonfinite = int((~torch.isfinite(cls_score)).sum().item())

        # s: foreground class probability matrix [N, M].
        cls_score = torch.nan_to_num(cls_score, nan=0.0, posinf=0.0, neginf=0.0)
        probs = F.softmax(cls_score, dim=1)[:, :-1]
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        s = probs[:, gt_labels]

        # u: IoU matrix between decoded bboxes and gts [N, M].
        u_raw = self._decode_iou_matrix(proposals, bbox_pred, gt_bboxes, gt_labels,
                                        bbox_coder)
        iou_nonfinite = int((~torch.isfinite(u_raw)).sum().item())
        u = u_raw
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        if self.candidate_iou_thr > 0:
            candidate_mask = u >= self.candidate_iou_thr
        else:
            candidate_mask = torch.ones_like(u, dtype=torch.bool)

        # t = s^alpha * u^beta.
        t = (s.clamp(min=self.eps)**self.alpha) * (u.clamp(min=self.eps)**self.beta)
        t = t * candidate_mask.to(dtype=t.dtype)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if num_gts > 0:
            candidate_per_gt = candidate_mask.sum(dim=0)
            candidate_stats = dict(
                mean=float(candidate_per_gt.float().mean().item()),
                min=int(candidate_per_gt.min().item()),
                max=int(candidate_per_gt.max().item()))
        else:
            candidate_stats = dict(mean=0.0, min=0, max=0)

        topk = min(self.topk, num_props)
        if topk < 1:
            return self._empty_result(num_gts, num_props, device)

        _, topk_inds = torch.topk(t, k=topk, dim=0, largest=True, sorted=True)

        t_hat_matrix = torch.zeros_like(t)
        for gt_idx in range(num_gts):
            pos_inds = topk_inds[:, gt_idx]
            pos_t = t[pos_inds, gt_idx]
            pos_u = u[pos_inds, gt_idx]

            max_t = pos_t.max().clamp(min=self.eps)
            max_u = pos_u.max().clamp(min=self.eps)
            t_hat = pos_t / max_t * max_u
            t_hat = torch.where(pos_t > 0, t_hat, torch.zeros_like(t_hat))
            t_hat_matrix[pos_inds, gt_idx] = t_hat

        assigned_gt_inds = torch.zeros(num_props, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_props, ), -1, dtype=torch.long, device=device)
        assigned_t = torch.zeros(num_props, dtype=torch.float, device=device)
        assigned_u = torch.zeros(num_props, dtype=torch.float, device=device)

        for prop_idx in range(num_props):
            matched_gts = torch.nonzero(
                t_hat_matrix[prop_idx] > 0, as_tuple=False).flatten()
            if matched_gts.numel() == 0:
                continue

            if matched_gts.numel() == 1:
                gt_idx = matched_gts.item()
            else:
                if self.use_max_t_when_conflict:
                    best_local = torch.argmax(t[prop_idx, matched_gts])
                else:
                    best_local = torch.argmax(t_hat_matrix[prop_idx, matched_gts])
                gt_idx = matched_gts[best_local].item()

            assigned_gt_inds[prop_idx] = gt_idx + 1
            assigned_labels[prop_idx] = gt_labels[gt_idx]
            assigned_t[prop_idx] = t[prop_idx, gt_idx]
            assigned_u[prop_idx] = u[prop_idx, gt_idx]

        assigned_t_hat = torch.zeros(num_props, dtype=torch.float, device=device)
        pos_mask = assigned_gt_inds > 0
        if pos_mask.any():
            pos_inds = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
            matched_gt_inds = assigned_gt_inds[pos_inds] - 1
            assigned_t_hat[pos_inds] = t_hat_matrix[pos_inds, matched_gt_inds]
            t_hat_pos_stats = self._tensor_stats(assigned_t_hat[pos_inds])
        else:
            t_hat_pos_stats = dict(mean=0.0, min=0.0, max=0.0)

        match_count = (t_hat_matrix > 0).sum(dim=1)
        multi_match_props = int((match_count > 1).sum().item())
        assigned_pos = int(pos_mask.sum().item())
        assigned_neg = int((assigned_gt_inds == 0).sum().item())

        assign_result = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=assigned_u.clone(),
            labels=assigned_labels)
        _set_assign_extra(assign_result, 'assign_metrics', assigned_t)
        _set_assign_extra(assign_result, 'assign_ious', assigned_u)
        _set_assign_extra(assign_result, 'assign_t_hat', assigned_t_hat)
        _set_assign_extra(
            assign_result, 'assign_debug',
            dict(
                num_props=int(num_props),
                num_gts=int(num_gts),
                cls_nonfinite=cls_nonfinite,
                iou_nonfinite=iou_nonfinite,
                s_stats=self._tensor_stats(s),
                u_stats=self._tensor_stats(u),
                t_stats=self._tensor_stats(t),
                candidate_per_gt=candidate_stats,
                multi_match_props=multi_match_props,
                assigned_pos=assigned_pos,
                assigned_neg=assigned_neg,
                t_hat_pos_stats=t_hat_pos_stats))
        return assign_result
