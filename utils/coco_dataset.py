import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.vision import VisionDataset


class Coco_Caption(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class Coco_Image(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        load_slice: Optional[str] = 'all',
        test: Optional[bool] = False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        slice_index = [0,55632,60000,90000,-1]
        self.ids = list(sorted(self.coco.imgs.keys()))
        if load_slice == 'all':
            self.ids = self.ids
        else:
            load_slice = int(load_slice)
            self.ids = self.ids[slice_index[load_slice-1]:slice_index[load_slice]]
        self.test = test

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    def load_target_super(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in self.load_target_super(id)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id) if not self.test else ['']

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target,self.coco.loadImgs(id)[0]["file_name"]

    def __len__(self) -> int:
        return len(self.ids)
class Coco_Image_Canny_path(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        canny_root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        load_slice: Optional[str] = 'all',
        test: Optional[bool] = False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.canny_root = canny_root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.image_transform = transform
        self.test = test
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    def _load_canny(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = path[:-4] + '.png'
        return Image.open(os.path.join(self.canny_root, path)).convert("RGB")
    def load_target_super(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in self.load_target_super(id)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id) if not self.test else ['']
        canny_image = self._load_canny(id)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        canny_image = self.image_transform(canny_image)
        return image, target,canny_image,self.coco.loadImgs(id)[0]["file_name"]

    def __len__(self) -> int:
        return len(self.ids)
class Coco_Image_Canny(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        canny_root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        load_slice: Optional[str] = 'all',
        test: Optional[bool] = False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.canny_root = canny_root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.image_transform = transform
        self.test = test
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    def _load_canny(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = path[:-4] + '.png'
        return Image.open(os.path.join(self.canny_root, path)).convert("RGB")
    def load_target_super(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in self.load_target_super(id)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id) if not self.test else ['']
        canny_image = self._load_canny(id)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        canny_image = self.image_transform(canny_image)
        return image, target,canny_image

    def __len__(self) -> int:
        return len(self.ids)


class Coco_Image_Canny_Uncertainty(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        canny_root: Union[str, Path],
        uncertainty_root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        load_slice: Optional[str] = 'all',
        test: Optional[bool] = False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.canny_root = canny_root
        self.uncertainty_root = uncertainty_root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.image_transform = transform
        self.test = test
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    def _load_canny(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = path[:-4] + '.png'
        return Image.open(os.path.join(self.canny_root, path)).convert("RGB")
    def _load_uncertainty(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = path[:-4] + '.png'
        return Image.open(os.path.join(self.uncertainty_root, path)).convert("RGB")
    def load_target_super(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in self.load_target_super(id)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        canny_target = self._load_canny(id)
        target = self._load_target(id) if not self.test else ['']
        canny_uncertainty = self._load_uncertainty(id)#self._load_canny(id)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        canny_target = self.image_transform(canny_target)
        canny_uncertainty = self.image_transform(canny_uncertainty)
        return image, target,canny_target,canny_uncertainty

    def __len__(self) -> int:
        return len(self.ids)
