_base_ = [
    './radcloud_original.py'
]

model = dict(
    backbone = dict(
        upsample_cfg=dict(
            type='InterpConv',
            _delete_=True
        )
    )
)