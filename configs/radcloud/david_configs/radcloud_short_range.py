_base_ = [
    './radcloud_original.py'
]

in_channels = 40

model = dict(
    backbone = dict(
        in_channels=in_channels
    )
)


train_data_root = '/data/david/train'
test_data_root = "/data/david/test"

train_dataloader = dict(
    dataset = dict(
        data_root = train_data_root
    )
)

val_dataloader = dict(
    dataset = dict(
        data_root=test_data_root
    )
)

test_dataloader = val_dataloader
