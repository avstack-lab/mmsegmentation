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
        classes=("point","background"),
        palette=[[1,1,1],[0,0,0]])

    def __init__(self,
                 seg_map_suffix='.png',
                 img_suffix=".jpg",
                 data_prefix=dict(
                     img_path="radar",
                     seg_map_path="lidar"
                 ),
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            img_suffix=img_suffix,
            data_prefix=data_prefix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        