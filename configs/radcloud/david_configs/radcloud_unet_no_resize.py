_base_ = [
    '../../_base_/models/fcn_unet_s5-d16.py',
    '../../_base_/datasets/radcloud.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]

crop_size = (64,48)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        scale=crop_size,
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=1.0, keep_ratio=True)
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size = 256,
    dataset = dict(
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size = 1,
    dataset=dict(
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

data_preprocessor = dict(
    size=crop_size,
    mean=None,
    std = None,
    bgr_to_rgb = False
)


decode_info = dict(
    out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0
    )
)
auxiliary_info = dict(
    out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.4
    )
)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg = dict(
        mode='whole'
    ),
    decode_head = decode_info,
    auxiliary_head = auxiliary_info,
)

train_cfg = dict(
    max_iters=2000,
    val_interval=500
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500)
)