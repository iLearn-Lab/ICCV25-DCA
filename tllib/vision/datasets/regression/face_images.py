"""
@author: Mingyang Liu
@contact: mingyangliu1024@gmail.com
"""
from typing import Optional, Sequence
import os
from typing import Optional, Callable, Tuple, Any, List, Sequence
from .._util import download as download_data, check_exits
from .image_regression import ImageRegression


class FaceImages(ImageRegression):
    """

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: APPA-REAL, \
            ``'U'``: UTKFace ,``'W'``: WIKI and ``'I'``: IMDB.
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        factors (sequence[str]): Factors selected. Default: ('age').
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

        """
   
    image_list = {
        "A": "APPA_REAL",
        "AC": "APPA_REAL_crop",
        "U": "UTKFace",
        "UC": "UTKFace_crop",
        "W": "wiki",
        "WC": "wiki_crop",
        "I": "imdb",
        "IC": "imdb_crop"
    }
    root_list = {
        "A": "APPA-REAL",
        "AC": "APPA-REAL",
        "U": "UTKFace",
        "UC": "UTKFace",
        "W": "IMDB-WIKI/WIKI",
        "WC": "IMDB-WIKI/WIKI",
        "I": "IMDB-WIKI/IMDB",
        "IC": "IMDB-WIKI/IMDB"
    }
    FACTORS = ('age',)

    def __init__(self, root: str, task: str, split: Optional[str] = 'train', mode="RGB",
                 factors: Sequence[str] = ('age',), download: Optional[bool] = False,
                 target_transform=None, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test']
        for factor in factors:
            assert factor in self.FACTORS

        factor_index = [self.FACTORS.index(factor) for factor in factors]

        if target_transform is None:
            target_transform = lambda x: x[list(factor_index)]
        else:
            target_transform = lambda x: target_transform(x[list(factor_index)])

        root = os.path.join(root, self.root_list[task])
        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(self.image_list[task], split))
        self.mode = mode

        super(FaceImages, self).__init__(root, factors, data_list_file=data_list_file, target_transform=target_transform, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[float]]:
        """
        Args:
            index (int): Index

        Returns:
            (image, target) where target is a numpy float array.
        """
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

