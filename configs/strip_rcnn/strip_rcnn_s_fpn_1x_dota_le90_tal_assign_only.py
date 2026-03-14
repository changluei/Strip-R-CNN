_base_ = ['./strip_rcnn_s_fpn_1x_dota_le90.py']

custom_imports = dict(
    imports=[
        'mmrotate.core.bbox.assigners.task_aligned_rroi_assigner',
        'mmrotate.models.roi_heads.task_aligned_oriented_standard_roi_head_assign_only',
    ],
    allow_failed_imports=False)

model = dict(
    roi_head=dict(
        type='TaskAlignedOrientedStandardRoIHeadAssignOnly',
        append_gt_as_proposals=True),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='TaskAlignedRRoIAssigner',
                topk=6,
                alpha=1.0,
                beta=6.0,
                candidate_iou_thr=0.0,
                use_max_t_when_conflict=True,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            pos_weight=-1,
            debug=False)))
