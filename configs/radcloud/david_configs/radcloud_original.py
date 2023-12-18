_base_ = [
    '../../_base_/models/radcloud.py',
    '../../_base_/datasets/radcloud.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]


default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=500,
        log_metric_by_epoch=False),
    checkpoint = dict(
        max_keep_ckpts=2,
        save_best='mAcc',
    )
)

model = dict(
    backbone = dict(
        norm_cfg = dict(
            type='BN'
        )
    ),
    decode_head = dict(
        norm_cfg = dict(
            type="BN"
        )
    ),
    auxiliary_head = dict(
        norm_cfg = dict(
            type = "BN"
        )
    )
)