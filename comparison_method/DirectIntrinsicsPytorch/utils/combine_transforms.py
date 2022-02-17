import torch
import random
import numpy as np
from torchvision.transforms.functional import normalize

from PIL import Image, ImageOps
import pdb


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_in = normalize(img_in, self.mean, self.std)
        img_bg = normalize(img_bg, self.mean, self.std)
        img_rf = normalize(img_rf, self.mean, self.std)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class ToTensor_v0(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_in = np.array(img_in).astype(np.float32).transpose((2, 0, 1))
        img_bg = np.array(img_bg).astype(np.float32).transpose((2, 0, 1))
        img_rf = np.array(img_rf).astype(np.float32).transpose((2, 0, 1))

        img_in = torch.from_numpy(img_in).float()
        img_bg = torch.from_numpy(img_bg).float()
        img_rf = torch.from_numpy(img_rf).float()

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_bg = np.array(img_bg).astype(np.float32) / 255.
        img_in = np.array(img_in).astype(np.float32) / 255.
        img_rf = np.array(img_rf).astype(np.float32) / 255.

        img_shape = img_in.shape

        if len(img_shape) == 3:
            img_in = img_in.transpose((2, 0, 1))
            img_bg = img_bg.transpose((2, 0, 1))
            img_rf = img_rf.transpose((2, 0, 1))

            img_in = torch.from_numpy(img_in).float()
            img_bg = torch.from_numpy(img_bg).float()
            img_rf = torch.from_numpy(img_rf).float()
        else:
            img_in = torch.from_numpy(img_in).float().unsqueeze(0)
            img_bg = torch.from_numpy(img_bg).float().unsqueeze(0)
            img_rf = torch.from_numpy(img_rf).float().unsqueeze(0)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        if random.random() < 0.5:
            img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
            img_bg = img_bg.transpose(Image.FLIP_LEFT_RIGHT)
            img_rf = img_rf.transpose(Image.FLIP_LEFT_RIGHT)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        rotate_degree = random.uniform(-1*self.degree, self.degree)

        img_in = img_in.rotate(rotate_degree, Image.BILINEAR)
        img_bg = img_bg.rotate(rotate_degree, Image.BILINEAR)
        img_rf = img_rf.rotate(rotate_degree, Image.BILINEAR)

        return {'input': img_in,
                'background': img_bg,
                'reflection': img_rf}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img_in = ImageOps.expand(img_in, border=(0, 0, padw, padh), fill=0)
            img_bg = ImageOps.expand(img_bg, border=(0, 0, padw, padh), fill=0)
            img_rf = ImageOps.expand(img_rf, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img_in.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomScaleCrop_refine(object):
    """ for MPI:
        1. crop without black edges: scale factor [0.8,1.2]
            shorter edge is 436, while crop window is 336. 336/436=0.7706
            [0.6, 1.5]for 256 pratch
        2. use mask images as crop reference
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.w = None
        self.h = None
        self.x1 = None
        self.y1 = None

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        img_mask = sample['M']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.78),
                                    int(self.base_size * 1.3))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        # check if the patch is valid
        resample_flag = True
        loop_n = 0
        while resample_flag:
            if loop_n >= 10 and loop_n % 10 == 0:
                print('resample valid patch...loop times:', loop_n)
            else:
                pass
            loop_n += 1
            img_mask = img_mask.resize((ow, oh), Image.NEAREST)
            self.w, self.h = img_mask.size
            self.x1 = random.randint(0, w - self.crop_size)
            self.y1 = random.randint(0, h - self.crop_size)
            patch_mask = img_mask.crop((self.x1, self.y1,
                                       self.x1 + self.crop_size, self.y1 + self.crop_size))
            mask_idx = np.array(patch_mask, dtype=np.float32) == 0
            mask_num = mask_idx.sum().astype(np.float32)
            resample_flag =  mask_num > 0.05 * mask_idx.size
        # pdb.set_trace()

        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        # pad crop
        # random crop crop_size
        #w, h = img_in.size
        #x1 = random.randint(0, w - self.crop_size)
        #y1 = random.randint(0, h - self.crop_size)
        x1 = self.x1
        y1 = self.y1
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomScaleCrop_refine_RD(object):
    """ for MPI:
        1. crop without black edges: scale factor [0.8,1.2]
            shorter edge is 436, while crop window is 336. 336/436=0.7706
            [0.6, 1.5]for 256 pratch
        2. use mask images as crop reference
        for MPI-RD: do not use masks
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.w = None
        self.h = None
        self.x1 = None
        self.y1 = None

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        img_mask = sample['M']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.9),
                                    int(self.base_size * 1.2))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        # check if the patch is valid
        resample_flag = False
        loop_n = 0
        while resample_flag:
            if loop_n >= 10 and loop_n % 10 == 0:
                print('resample valid patch...loop times:', loop_n)
            else:
                pass
            loop_n += 1
            img_mask = img_mask.resize((ow, oh), Image.NEAREST)
            self.w, self.h = img_mask.size
            self.x1 = random.randint(0, w - self.crop_size)
            self.y1 = random.randint(0, h - self.crop_size)
            patch_mask = img_mask.crop((self.x1, self.y1,
                                       self.x1 + self.crop_size, self.y1 + self.crop_size))
            mask_idx = np.array(patch_mask, dtype=np.float32) == 0
            mask_num = mask_idx.sum().astype(np.float32)
            resample_flag =  mask_num > 0.05 * mask_idx.size
        # pdb.set_trace()

        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        # pad crop
        # random crop crop_size
        w, h = img_in.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        w, h = img_in.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img_in.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_in = img_in.resize(self.size, Image.BILINEAR)
        img_bg = img_bg.resize(self.size, Image.BILINEAR)
        img_rf = img_rf.resize(self.size, Image.BILINEAR)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class ScaleWidth(object):
    def __init__(self, size):
        self.target_width = size

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        oh, ow = img_in.size
        if ow == self.target_width:
            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf}

        w = self.target_width
        h = int(self.target_width * oh / ow)

        img_in = img_in.resize((w, h), Image.BICUBIC)
        img_bg = img_bg.resize((w, h), Image.BICUBIC)
        img_rf = img_rf.resize((w, h), Image.BICUBIC)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedRescale(object):
    def __init__(self, scale):
        self.scale = scale  # scale: float

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        w, h = img_in.size
        ow = int(w * self.scale)
        oh = int(h * self.scale)
        if ow > 980 and ow < 1024:
            ow = 992
            oh = 416
        else:
            #oh = 224
            pass
        self.size = (ow, oh)
        img_in = img_in.resize(self.size, Image.BILINEAR)
        img_bg = img_bg.resize(self.size, Image.BILINEAR)
        img_rf = img_rf.resize(self.size, Image.BILINEAR)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedScalePadding(object):
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        w, h = img_in.size
        if h < w:
            ow = self.size
            oh = int(1.0 * h * ow / w)
        else:
            oh = self.size
            ow = int(1.0 * w * oh / h)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        pad_h = self.size - oh if oh < self.size else 0
        pad_w = self.size - ow if ow < self.size else 0

        img_in = ImageOps.expand(img_in, border=(0, 0, pad_w, pad_h), fill=0)
        img_bg = ImageOps.expand(img_bg, border=(0, 0, pad_w, pad_h), fill=0)
        img_rf = ImageOps.expand(img_rf, border=(0, 0, pad_w, pad_h), fill=0)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}
