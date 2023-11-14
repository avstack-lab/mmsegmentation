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
    dataset = val_dataloader
)