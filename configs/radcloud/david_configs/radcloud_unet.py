_base_ = [
    '../../_base_/models/deeplabv3_unet_s5-d16.py',
    '../../_base_/datasets/radcloud.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]

crop_size = (64,64)
img_scale = (100,100)
train_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='LoadAnnotationsNP'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotationsNP'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromNP', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotationsNP')], [dict(type='PackSegInputs')]
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

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg = dict(
        crop_size=crop_size,
        stride=(42,42)
    )
)

train_cfg = dict(
    max_iters=2000,
    val_interval=500
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500)
)