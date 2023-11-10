_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/radcloud.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type="BN",requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2,norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2,norm_cfg=norm_cfg),
    pretrained=None,
    backbone=dict(depth=101,norm_cfg=norm_cfg))
