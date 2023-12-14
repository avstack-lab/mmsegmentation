_base_ = [
    './radcloud_original_rand_flip.py'
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

#train for 40k iters
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]

# training schedule for 40k
train_cfg = dict(
    max_iters=40000,
    val_interval=4000
)

default_hooks = dict(
    checkpoint=dict(
        interval=4000
    )
)