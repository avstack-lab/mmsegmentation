_base_ = [
    '../../_base_/models/radcloud.py',
    '../../_base_/datasets/radcloud.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]

train_cfg = dict(
    max_iters=5000,
    val_interval=1000
)

model = dict(
    decode_head = dict(
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        out_channels=2
    )
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500)
)