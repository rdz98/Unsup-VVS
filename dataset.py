import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class STL10(torchvision.datasets.STL10):
    def __init__(self, rp_transform=None, biaugment=False, *args, **kwargs):
        self.rp_transform = rp_transform
        self.biaugment = biaugment
        super(STL10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target1 = self.data[index], int(self.labels[index])
        else:
            img, target1 = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = transforms.ToTensor()(pil_img)

        if self.transform is not None:
            img1 = self.transform(pil_img)
            if self.biaugment:
                img2 = self.transform(pil_img)
        else:
            img1 = pil_img
            if self.biaugment:
                img2 = pil_img

        if self.target_transform is not None:
            target1 = self.target_transform(target1)

        (img3, img4), target2 = get_relative_position_pair(img)
        if self.rp_transform:
            img3 = self.rp_transform(img3)
            img4 = self.rp_transform(img4)

        if self.biaugment:
            return (img1, img2, img3, img4), (target1, target2), index
        else:
            return (img1, img3, img4), (target1, target2), index


def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass


def _get_quarter_block(img, idx):
    origin_size = img.shape[-1]
    crop_size = origin_size // 2
    if idx == 0:
        crop_img = img[:, :crop_size, :crop_size]
    elif idx == 1:
        crop_img = img[:, crop_size:, :crop_size]
    elif idx == 2:
        crop_img = img[:, :crop_size, crop_size:]
    else:
        crop_img = img[:, crop_size:, crop_size:]
    return transforms.Resize(origin_size)(crop_img)


def get_relative_position_pair(img):
    a, b = np.random.choice(np.arange(4), 2, replace=False)
    target_dict = {
        (0, 1): 2,
        (0, 2): 4,
        (0, 3): 3,
        (1, 0): 6,
        (1, 2): 5,
        (1, 3): 4,
        (2, 0): 0,
        (2, 1): 1,
        (2, 3): 2,
        (3, 0): 7,
        (3, 1): 0,
        (3, 2): 6
    }
    img1 = _get_quarter_block(img, a)
    img2 = _get_quarter_block(img, b)
    target = target_dict[a, b]
    return (img1, img2), target
