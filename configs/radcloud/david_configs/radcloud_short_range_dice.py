_base_ = [
    './radcloud_short_range.py'
]

model = dict(
    decode_head = dict(
        loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=True, loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', use_sigmoid=True,loss_name='loss_dice', loss_weight=3.0)
        ]
    )
)