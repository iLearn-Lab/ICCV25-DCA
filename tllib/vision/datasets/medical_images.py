"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import os
from typing import Optional, Tuple, Any

import torch


from .imagelist import ImageList
from ._util import download as download_data, check_exits
from typing import Optional, Callable, Tuple, Any, List, Iterable


class NIH_CXR14(ImageList):

    image_list = {
        "train": "image_list/NIH_CXR14_train.txt",
        "test": "image_list/NIH_CXR14_test.txt",
        "train_without_normal": "image_list/NIH_CXR14_train_without_normal.txt",
        "test_without_normal": "image_list/NIH_CXR14_test_without_normal.txt"
    }
    classes = ['0 - Atelectasis', '1 - Cardiomegaly', '2 - Effusion', '3 - Consolidation', '4 - Edema',
               '5 - Pneumonia', '6 - Normal']
    def __init__(self, root, mode="RGB", split='train', without_normal=False,download: Optional[bool] = True, **kwargs):
        assert split in ['train', 'test', 'train_without_normal','test_without_normal']
        data_list_file = os.path.join(root, self.image_list[split])
        assert mode in ['L', 'RGB']
        self.mode = mode

        super(NIH_CXR14, self).__init__(root, NIH_CXR14.get_classes(without_normal), data_list_file=data_list_file, **kwargs, multi_labels=True)

    def __getitem__(self, index: int) -> Tuple[Any, List[int]]:
        """
        Args:
            index (int): Index

        return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        # print("index:"+str(index))
        # print(path)
        # print(target)
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        # 一般target_transform都是None
        if self.target_transform is not None and target is not None:
            label = self.target_transform(target)
        # 升维，变成1*cls_num，这样Dataloader才知道这个一个image的label；否则会以为是cls_num个label
        label = torch.IntTensor(target).view([1, -1])
        return img, label

    @classmethod
    def get_classes(self,without_normal):
        if without_normal:
            return NIH_CXR14.classes[0:-1]
        else:
            return NIH_CXR14.classes

class Open_i(ImageList):

    image_list = {
        "train": "image_list/Open_i_train.txt",
        "test": "image_list/Open_i_test.txt",
        "train_without_normal": "image_list/Open_i_train_without_normal.txt",
        "test_without_normal": "image_list/Open_i_test_without_normal.txt"
    }
    classes = ['0 - Atelectasis', '1 - Cardiomegaly', '2 - Effusion', '3 - Consolidation', '4 - Edema',
               '5 - Pneumonia', '6 - Normal']
    def __init__(self, root, mode="RGB", split='train', without_normal=False, download: Optional[bool] = True, **kwargs):
        assert split in ['train', 'test','train_without_normal', 'test_without_normal']
        data_list_file = os.path.join(root, self.image_list[split])
        assert mode in ['L', 'RGB']
        self.mode = mode
        super(Open_i, self).__init__(root, Open_i.get_classes(without_normal), data_list_file=data_list_file, **kwargs, multi_labels=True)

    def __getitem__(self, index: int) -> Tuple[Any, List[int]]:
        """
        Args:
            index (int): Index

        return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        # 一般target_transform都是None
        if self.target_transform is not None and target is not None:
            label = self.target_transform(target)
        # 升维，变成1*cls_num，这样Dataloader才知道这个一个image的label；否则会以为是cls_num个label
        label = torch.IntTensor(target).view([1, -1])
        return img, label

    @classmethod
    def get_classes(self, without_normal):
        if without_normal:
            return Open_i.classes[0:-1]
        else:
            return Open_i.classes



class CheXpert(ImageList):

    image_list = {
        "train": "image_list/CheXpert_train.txt",
        "test": "image_list/CheXpert_test.txt",
        "train_resized": "image_list/CheXpert_train_resized.txt",
        "test_resized": "image_list/CheXpert_test_resized.txt",
        "train_without_normal": "image_list/CheXpert_train_without_normal.txt",
        "test_without_normal": "image_list/CheXpert_test_without_normal.txt"
    }
    classes = ['0 - Atelectasis', '1 - Cardiomegaly', '2 - Effusion', '3 - Consolidation', '4 - Edema',
               '5 - Pneumonia', '6 - Normal']
    def __init__(self, root, mode="RGB", split='train', without_normal=False,download: Optional[bool] = True, **kwargs):
        assert split in ['train', 'test', 'train_resized', 'test_resized', 'train_without_normal', 'test_without_normal']
        data_list_file = os.path.join(root, self.image_list[split])
        assert mode in ['L', 'RGB']
        self.mode = mode
        super(CheXpert, self).__init__(root, CheXpert.get_classes(without_normal), data_list_file=data_list_file, **kwargs, multi_labels=True)

    def __getitem__(self, index: int) -> Tuple[Any, List[int]]:
        """
        Args:
            index (int): Index

        return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        # 一般target_transform都是None
        if self.target_transform is not None and target is not None:
            label = self.target_transform(target)
        # 升维，变成1*cls_num，这样Dataloader才知道这个一个image的label；否则会以为是cls_num个label
        label = torch.IntTensor(target).view([1, -1])
        return img, label

    @classmethod
    def get_classes(self, without_normal):
        if without_normal:
            return CheXpert.classes[0:-1]
        else:
            return CheXpert.classes
        

# single label medical images
class MedicalImages(ImageList):

    image_list = {
        "N": "NIH_CXR14",
        "C": "CheXpert",
        "M": "MIMIC_CXR",
        "O": "Open_i",
    }
    root_list = {
        "N": "NIH_CXR14",
        "C": "CheXpert",
        "M": "MIMIC_CXR",
        "O": "Open_i",
    }
     
    classes = ['0 - N', '1 - P']
    def __init__(self, root, task, class_index, split='train', download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test', 'uncertain', 'train_without_uncertain', 'test_without_uncertain']
        data_list_file_path = os.path.join(root, self.root_list[task], 'image_list', '{}_{}_{}.txt'.format(self.image_list[task],str(class_index),split))
        super(MedicalImages, self).__init__(root, MedicalImages.get_classes(), data_list_file=data_list_file_path, **kwargs)


    @classmethod
    def get_classes(self):
        return self.classes