_base_ = [
    "../../unet/unet-s5-d16_deeplabv3_4xb4-40k_drive-64x64.py"
]

data_root = '/data/david/Drive_CPSL/'
val_dataloader = dict(
    dataset = dict(
        data_root=data_root,
        seg_map_suffix='.png'
    )
)
test_dataloader = val_dataloader

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type='DRIVEDataset',
        data_root='/data/david/Drive_CPSL/',
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='RandomResize',
                    scale=(584, 565),
                    ratio_range=(0.5, 2.0),
                    keep_ratio=True),
                dict(
                    type='RandomCrop', crop_size=(64, 64), cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(type='PackSegInputs')
            ],
        seg_map_suffix='.png'),
        _delete_=True)