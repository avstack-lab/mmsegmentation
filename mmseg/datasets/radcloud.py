# Xiao 2023-10-04
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class RadcloudDataset(BaseSegDataset):
    """RadcloudDataset dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    METAINFO = dict(
        classes=("background","point"),
        palette=[[0,0,0],[1,1,1]])

    def __init__(self, **kwargs) -> None:
        super().__init__(
            seg_map_suffix=".npy",
            img_suffix=".npy",
            data_prefix=dict(
                     img_path="radar",
                     seg_map_path="lidar"
                 ),
            reduce_zero_label=False,
            **kwargs)
        