_base_ = ['./rotated_retinanet_obb_r50_fpn_1x_dota_le90.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='StripNet',
        embed_dims=[32, 64, 160, 256],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        drop_rate=0.1,
        drop_path_rate=0.15,
        depths=[3, 3, 5, 2],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/stripnet_t.pth'),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        in_channels=[32, 64, 160, 256],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05)
