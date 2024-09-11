from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FieldOfViewBevDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("invisible", "visible"),
        palette=[[128, 64, 128], [10, 120, 232]],
    )

    def __init__(
        self,
        img_suffix='.png',
        seg_map_suffix='_segimage.png',
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
