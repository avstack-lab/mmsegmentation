_base_ = [
    '../../_base_/models/fast_scnn.py',
    '../../_base_/datasets/radcloud.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]

crop_size = (512,1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    size= crop_size,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    data_preprocessor = data_preprocessor
)

train_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='LoadAnnotationsNP'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotationsNP'),
    dict(type='PackSegInputs')
]
tta_pipeline = [
    dict(type='LoadImageFromNP', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            dict(type='Resize', scale=crop_size, keep_ratio=True),
            dict(type='LoadAnnotationsNP'),
            dict(type='PackSegInputs')
        ])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

optimizer = dict(
    type='SGD',
    lr=0.12,
    momentum=0.9,
    weight_decay=4e-5
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer
)

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=2000,
        log_metric_by_epoch=False),
    checkpoint = dict(
        max_keep_ckpts=2,
        save_best='mAcc',
    )
)