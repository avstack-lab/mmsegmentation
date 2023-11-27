_base_ = [
    './radcloud_original.py'
]

crop_size = (64, 48)
train_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='LoadAnnotationsNP'),
    # dict(
    #     type='RandomResize',
    #     scale=img_scale,
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction="horizontal"),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromNP'),
    # dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotationsNP'),
    dict(type='PackSegInputs')
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromNP', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            # [
            #     dict(type='Resize', scale_factor=r, keep_ratio=True)
            #     for r in img_ratios
            # ],
            # [
            #     dict(type='RandomFlip', prob=0., direction='horizontal'),
            #     dict(type='RandomFlip', prob=1., direction='horizontal')
            # ], 
            [
                dict(type='LoadAnnotationsNP')
            ],
            [
                dict(type='PackSegInputs')
            ]
        ])
]

train_dataloader = dict(
    dataset = dict(
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    dataset = dict(
        pipeline=test_pipeline
    )
)
