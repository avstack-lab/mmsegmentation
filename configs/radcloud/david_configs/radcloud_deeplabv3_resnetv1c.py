_base_ = [
    './radcloud_original.py'
]
#deeplab v3 base config from deeplabv3_unet_s5-d16.py

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone = dict(
        upsample_cfg=dict(
            type='InterpConv',
            _delete_=True
        )
    ),
    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=3,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        out_channels=1,
        _delete_=True
    )
)

model = dict(
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        _delete_=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        out_channels=1,
        _delete_=True
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256
    )
)
