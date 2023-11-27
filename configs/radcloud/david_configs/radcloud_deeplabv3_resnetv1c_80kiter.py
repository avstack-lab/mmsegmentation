_base_ = [
    './radcloud_deeplabv3_resnetv1c.py'
]

train_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='LoadAnnotationsNP'),
    dict(type='RandomFlip', prob=0.5, direction="horizontal"),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='LoadAnnotationsNP'),
    dict(type='PackSegInputs')
]
tta_pipeline = [
    dict(type='LoadImageFromNP', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
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



param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False)
]

train_cfg = dict(max_iters=80000)
resume_from = "../../../work_dirs/radcloud_deeplabv3_resnetv1c/iter_20000.pth"