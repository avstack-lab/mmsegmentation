norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    size=(64, 48),
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        size=(64, 48),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=4,
        strides=(1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2),
        dec_num_convs=(2, 2, 2),
        downsamples=(True, True, True),
        enc_dilations=(1, 1, 1, 1),
        dec_dilations=(1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='DeconvModule', kernel_size=2),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        out_channels=1),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
        out_channels=1),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'RadcloudDataset'
train_data_root = '/data/david/CPSL_Ground/train'
test_data_root = '/data/david/CPSL_Ground/test'
train_pipeline = [
    dict(type='LoadImageFromNP'),
    dict(type='LoadAnnotationsNP'),
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
        transforms=[[{
            'type': 'LoadAnnotationsNP'
        }], [{
            'type': 'PackSegInputs'
        }]])
]
train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RadcloudDataset',
        data_root='/data/david/CPSL_Ground/train',
        pipeline=[
            dict(type='LoadImageFromNP'),
            dict(type='LoadAnnotationsNP'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RadcloudDataset',
        data_root='/data/david/CPSL_Ground/test',
        pipeline=[
            dict(type='LoadImageFromNP'),
            dict(type='LoadAnnotationsNP'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RadcloudDataset',
        data_root='/data/david/CPSL_Ground/test',
        pipeline=[
            dict(type='LoadImageFromNP'),
            dict(type='LoadAnnotationsNP'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
work_dir = '../submodules/lib-avstack-core/third_party/mmsegmentation/work_dirs/playground'
