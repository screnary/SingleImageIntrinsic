import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import os.path
from utils import combine_transforms as ctr
from PIL import Image
from torchvision import transforms
from data.data_loader import CreateDataLoaderIIW
from data.data_loader import CreateDataLoaderIIWTest
from data.data_loader import CreateDataLoaderSAW
import pdb


###############################################################################
# Dataset sets
###############################################################################


class DatasetIdMIT(data.Dataset):
    def __init__(self, data_opt, is_train=None):
        self.opt = data_opt
        self.root = data_opt.data_root
        self.is_train = data_opt.is_train if is_train is None else is_train

        self.I_paths = []
        self.B_paths = []
        self.R_paths = []

        root_i = os.path.join(self.root, 'MIT', 'MIT-input')
        root_r = os.path.join(self.root, 'MIT', 'MIT-reflectance')
        root_b = os.path.join(self.root, 'MIT', 'MIT-shading')

        root_i_fs = os.path.join(self.root, 'MIT', 'MIT-input-fullsize')
        root_r_fs = os.path.join(self.root, 'MIT', 'MIT-reflectance-fullsize')
        root_b_fs = os.path.join(self.root, 'MIT', 'MIT-shading-fullsize')

        fname = 'train.txt' if self.is_train else 'test.txt'

        with open(os.path.join(self.root, 'MIT', fname), 'r') as fid:
            lines = fid.readlines()
            if is_train:
                for line in lines:
                    line = line.strip()
                    self.I_paths.append(os.path.join(root_i_fs, line))
                    self.B_paths.append(os.path.join(root_b_fs, line))
                    self.R_paths.append(os.path.join(root_r_fs, line))

                self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size, is_train=self.is_train,
                                                       no_flip=data_opt.no_flip,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)
            else:
                for line in lines:
                    line = line.strip()
                    self.I_paths.append(os.path.join(root_i_fs, line))
                    self.B_paths.append(os.path.join(root_b_fs, line))
                    self.R_paths.append(os.path.join(root_r_fs, line))

                self.transform = get_combine_transform(name='none', load_size=data_opt.load_size,
                                                       new_size=data_opt.load_size, is_train=self.is_train,
                                                       no_flip=True,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)

    def __len__(self):
        return len(self.I_paths)

    def _transform_image(self, image, seed=None):
        if self.transform is not None:
            seed = np.random.randint(100, 500000) if seed is None else seed
            torch.manual_seed(seed)
            random.seed = seed
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        seed = np.random.randint(100, 500000)
        if not self.opt.serial_batches:
            if self.opt.unpaired:
                # unpaired
                random.shuffle(self.I_paths)
                random.shuffle(self.B_paths)
                random.shuffle(self.R_paths)
            else:
                # paired
                random.shuffle(self.I_paths)
                self.B_paths = self.I_paths
                self.R_paths = self.I_paths
            seed = None

        ret_dict = {
            'I': Image.open(self.I_paths[index]).convert('RGB'),
            'B': Image.open(self.B_paths[index]).convert('RGB'),
            'R': Image.open(self.R_paths[index]).convert('RGB'),
        }

        ret_dict = self.transform(ret_dict)

        ret_dict['name'] = os.path.join(*self.I_paths[index].split('/')[-2:])

        return ret_dict


class DatasetIdMPI(data.Dataset):
    def __init__(self, data_opt, is_train=None, cropped=None):
        self.opt = data_opt
        self.root = data_opt.data_root
        self.is_train = data_opt.is_train if is_train is None else is_train
        self.is_cropped = data_opt.cropped if cropped is None else cropped

        self.I_paths = []
        self.B_paths = []
        self.R_paths = []
        if data_opt.name == 'MPI-auxilliary':
            root_i = os.path.join(self.root, 'MPI', 'MPI-auxilliary-input')
            root_r = os.path.join(self.root, 'MPI', 'MPI-auxilliary-albedo')
            root_b = os.path.join(self.root, 'MPI', 'MPI-auxilliary-shading')
            fname = 'train.txt' if self.is_train else 'test.txt'
        elif data_opt.name == 'MPI-main':
            if self.is_cropped:
                root_i = os.path.join(self.root, 'MPI', 'MPI-main-input-300')
                root_r = os.path.join(self.root, 'MPI', 'MPI-main-albedo-300')
                root_b = os.path.join(self.root, 'MPI', 'MPI-main-shading-300')

                fname = 'MPI_main_'+data_opt.split+'-300-train.txt' \
                    if self.is_train else 'MPI_main_'+data_opt.split+'-300-test.txt'
            else:
                root_i = os.path.join(self.root, 'MPI', 'MPI-main-clean')
                root_r = os.path.join(self.root, 'MPI', 'MPI-main-albedo')
                root_b = os.path.join(self.root, 'MPI', 'MPI-main-shading')
                if data_opt.split == 'imageSplit':
                    fname = 'MPI_main_imageSplit-fullsize-ChenSplit-train.txt' \
                        if self.is_train else 'MPI_main_imageSplit-fullsize-ChenSplit-test.txt'
                else:
                    fname = 'MPI_main_sceneSplit-fullsize-NoDefect-train.txt' \
                        if self.is_train else 'MPI_main_sceneSplit-fullsize-NoDefect-test.txt'
        else:
            raise NotImplementedError

        with open(os.path.join(self.root, 'MPI', fname), 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip()
                self.I_paths.append(os.path.join(root_i, line))
                self.B_paths.append(os.path.join(root_b, line))
                self.R_paths.append(os.path.join(root_r, line))

        if self.is_train:
            if not self.is_cropped:
                self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size, is_train=self.is_train,
                                                       no_flip=data_opt.no_flip,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)
            else:
                self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size, is_train=self.is_train,
                                                       no_flip=data_opt.no_flip,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)  # name='scale_width'
        else:
            if not self.is_cropped:
                self.transform = fixed_scale_transform(new_size=1.0, image_mean=data_opt.image_mean,
                                                       image_std=data_opt.image_std, use_norm=data_opt.use_norm)
            else:
                self.transform = get_combine_transform(name='scale_width', load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size_test, is_train=self.is_train,
                                                       no_flip=True,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)  # name='scale_width'

    def __len__(self):
        return len(self.I_paths)

    def _transform_image(self, image, seed=None):
        if self.transform is not None:
            seed = np.random.randint(100, 500000) if seed is None else seed
            torch.manual_seed(seed)
            random.seed = seed
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        seed = np.random.randint(100, 500000)
        if not self.opt.serial_batches:
            if self.opt.unpaired:
                # unpaired
                random.shuffle(self.I_paths)
                random.shuffle(self.B_paths)
                random.shuffle(self.R_paths)
            else:
                # paired
                idx = np.arange(len(self.I_paths))
                random.shuffle(idx)
                # pdb.set_trace()
                self.I_paths = np.array(self.I_paths)[idx]
                self.B_paths = np.array(self.B_paths)[idx]
                self.R_paths = np.array(self.R_paths)[idx]
            seed = None

        ret_dict = {
            'I': Image.open(self.I_paths[index]).convert('RGB'),
            'B': Image.open(self.B_paths[index]).convert('RGB'),
            'R': Image.open(self.R_paths[index]).convert('RGB'),
        }

        ret_dict = self.transform(ret_dict)

        ret_dict['name'] = os.path.join(*self.I_paths[index].split('/')[-2:])

        return ret_dict

class DatasetIdMPI_mask(data.Dataset):
    def __init__(self, data_opt, is_train=None, cropped=None):
        self.opt = data_opt
        self.root = data_opt.data_root
        self.is_train = data_opt.is_train if is_train is None else is_train
        self.is_cropped = data_opt.cropped if cropped is None else cropped

        self.I_paths = []
        self.B_paths = []
        self.R_paths = []
        self.M_paths = []
        if data_opt.name == 'MPI-auxilliary':
            root_i = os.path.join(self.root, 'MPI', 'MPI-auxilliary-input')
            root_r = os.path.join(self.root, 'MPI', 'MPI-auxilliary-albedo')
            root_b = os.path.join(self.root, 'MPI', 'MPI-auxilliary-shading')
            fname = 'train.txt' if self.is_train else 'test.txt'
        elif data_opt.name == 'MPI-main':
            if self.is_cropped:
                root_i = os.path.join(self.root, 'MPI', 'MPI-main-input-300')
                root_r = os.path.join(self.root, 'MPI', 'MPI-main-albedo-300')
                root_b = os.path.join(self.root, 'MPI', 'MPI-main-shading-300')

                fname = 'MPI_main_'+data_opt.split+'-300-train.txt' \
                    if self.is_train else 'MPI_main_'+data_opt.split+'-300-test.txt'
            else:
                root_i = os.path.join(self.root, 'MPI', 'MPI-main-clean')
                root_r = os.path.join(self.root, 'MPI', 'MPI-main-albedo')
                root_b = os.path.join(self.root, 'MPI', 'MPI-main-shading')
                root_m = os.path.join(self.root, 'MPI', 'MPI-main-mask')
                if data_opt.split == 'imageSplit':
                    fname = 'MPI_main_imageSplit-fullsize-ChenSplit-train.txt' \
                        if self.is_train else 'MPI_main_imageSplit-fullsize-ChenSplit-test.txt'
                else:
                    fname = 'MPI_main_sceneSplit-fullsize-NoDefect-train.txt' \
                        if self.is_train else 'MPI_main_sceneSplit-fullsize-NoDefect-test.txt'
        else:
            raise NotImplementedError

        with open(os.path.join(self.root, 'MPI', fname), 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip()
                self.I_paths.append(os.path.join(root_i, line))
                self.B_paths.append(os.path.join(root_b, line))
                self.R_paths.append(os.path.join(root_r, line))
                self.M_paths.append(os.path.join(root_m, line))

        if self.is_train:
            if not self.is_cropped:
                self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size, is_train=self.is_train,
                                                       no_flip=data_opt.no_flip,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)
            else:
                self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size, is_train=self.is_train,
                                                       no_flip=data_opt.no_flip,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)  # name='scale_width'
        else:
            if not self.is_cropped:
                self.transform = fixed_scale_transform(new_size=0.75, image_mean=data_opt.image_mean,
                                                       image_std=data_opt.image_std, use_norm=data_opt.use_norm)
            else:
                self.transform = get_combine_transform(name='scale_width', load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size_test, is_train=self.is_train,
                                                       no_flip=True,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)  # name='scale_width'

    def __len__(self):
        return len(self.I_paths)

    def _transform_image(self, image, seed=None):
        if self.transform is not None:
            seed = np.random.randint(100, 500000) if seed is None else seed
            torch.manual_seed(seed)
            random.seed = seed
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        seed = np.random.randint(100, 500000)
        if not self.opt.serial_batches:
            if self.opt.unpaired:
                # unpaired
                random.shuffle(self.I_paths)
                random.shuffle(self.B_paths)
                random.shuffle(self.R_paths)
            else:
                # paired
                idx = np.arange(len(self.I_paths))
                random.shuffle(idx)
                # pdb.set_trace()
                self.I_paths = np.array(self.I_paths)[idx]
                self.B_paths = np.array(self.B_paths)[idx]
                self.R_paths = np.array(self.R_paths)[idx]
                self.M_paths = np.array(self.M_paths)[idx]
            seed = None

        ret_dict_tmp = {
            'I': Image.open(self.I_paths[index]).convert('RGB'),
            'B': Image.open(self.B_paths[index]).convert('RGB'),
            'R': Image.open(self.R_paths[index]).convert('RGB'),
            'M': Image.open(self.M_paths[index]).convert('1')
        }

        ret_dict = self.transform(ret_dict_tmp)

        ret_dict['name'] = os.path.join(*self.I_paths[index].split('/')[-2:])

        return ret_dict


class DatasetIdIIW():
    def __init__(self, data_opt, is_train=None, mode=0, batch_size=None, is_check=False):
        self.is_train = data_opt.is_train if is_train is None else is_train
        self.batch_size = data_opt.batch_size if batch_size is None else batch_size
        self.root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.train_list_IIW = self.root + '/datasets/IIW/train_list/'
        self.test_list_IIW = self.root + '/datasets/IIW/test_list/'
        if self.is_train:
            # '/home/wzj/ws/Intrinsic/intrinsic-project/datasets/IIW'
            self.data_loader_IIW = CreateDataLoaderIIW(self.root, self.train_list_IIW, mode, self.batch_size)
            self.dataset_IIW = self.data_loader_IIW.load_data()
        else:
            self.data_loader_IIW = CreateDataLoaderIIWTest(self.root, self.test_list_IIW, mode, self.batch_size)
            self.dataset_IIW = self.data_loader_IIW.load_data()
        if is_check:
            self.data_loader_IIW = CreateDataLoaderIIWTest(self.root, self.train_list_IIW, mode, self.batch_size)
            self.dataset_IIW = self.data_loader_IIW.load_data()
        self.dataset_size_IIW = len(self.data_loader_IIW)

    def __len__(self):
        return self.dataset_size_IIW

    def load(self):
        return self.dataset_IIW


class DatasetIdSAW():
    def __init__(self, data_opt, is_train=None, mode=0, batch_size=None, is_check=False):
        self.is_train = data_opt.is_train if is_train is None else is_train
        self.batch_size = data_opt.batch_size if batch_size is None else batch_size
        self.root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.train_list_IIW = self.root + '/datasets/SAW/train_list/'
        self.test_list_IIW = self.root + '/datasets/SAW/train_list/'
        if self.is_train:
            # '/home/wzj/ws/Intrinsic/intrinsic-project/datasets/IIW'
            self.data_loader_IIW = CreateDataLoaderSAW(self.root, self.train_list_IIW, mode)
            self.dataset_IIW = self.data_loader_IIW.load_data()
        else:
            self.data_loader_IIW = CreateDataLoaderSAW(self.root, self.train_list_IIW, mode)
            self.dataset_IIW = self.data_loader_IIW.load_data()
        if is_check:
            self.data_loader_IIW = CreateDataLoaderSAW(self.root, self.train_list_IIW, mode)
            self.dataset_IIW = self.data_loader_IIW.load_data()
        self.dataset_size_IIW = len(self.data_loader_IIW)

    def __len__(self):
        return self.dataset_size_IIW

    def load(self):
        return self.dataset_IIW


###############################################################################
# Fast functions
###############################################################################

def get_combine_transform(name, load_size=300, new_size=256, is_train=True, no_flip=False,
                          image_mean=(0., 0., 0.), image_std=(1.0, 1.0, 1.0), use_norm=True):
    transform_list = []
    if name == 'resize_and_crop_refine':
        transform_list.append(ctr.RandomScaleCrop_refine(load_size, new_size))
    elif name == 'resize_and_crop':
        # o_size = [load_size, load_size]
        # transform_list.append(transforms.Scale(o_size, Image.BICUBIC))
        transform_list.append(ctr.RandomScaleCrop(load_size, new_size))
    elif name == 'crop':
        transform_list.append(ctr.RandomScaleCrop(load_size, new_size))
    elif name == 'scale_width':
        transform_list.append(ctr.ScaleWidth(new_size))
    elif name == 'scale_width_and_crop':
        transform_list.append(ctr.ScaleWidth(load_size))
        transform_list.append(transforms.RandomCrop(new_size))

    if is_train and not no_flip:
        transform_list.append(ctr.RandomHorizontalFlip())

    transform_list += [ctr.ToTensor()]

    if use_norm:
        transform_list += [ctr.Normalize(image_mean, image_std)]
    return transforms.Compose(transform_list)


def fixed_scale_transform(new_size=0.45,
                          image_mean=(0., 0., 0.), image_std=(1., 1., 1.), use_norm=True):
    transform_list = []
    # transform_list.append(ctr.FixedScalePadding(size=new_size))
    transform_list.append(ctr.FixedRescale(scale=new_size))
    transform_list += [ctr.ToTensor()]
    if use_norm:
        transform_list += [ctr.Normalize(image_mean, image_std)]

    return transforms.Compose(transform_list)


def test_id_dataset():
    from configs.intrinsic_mpi_v7 import opt
    from utils import tensor2img, show_image
    dataset = DatasetIdMIT(opt.data) if opt.data.name == 'MIT' else DatasetIdMPI_mask(opt.data)
    for i in range(len(dataset)):
        data = dataset[i]
        img_i, img_b, img_r = data['I'], data['B'], data['R']
        img_i = tensor2img(img_i, mean=opt.data.image_mean, std=opt.data.image_std)
        img_b = tensor2img(img_b, mean=opt.data.image_mean, std=opt.data.image_std)
        img_r = tensor2img(img_r, mean=opt.data.image_mean, std=opt.data.image_std)
        if not show_image(cv2.hconcat([img_i[:, :, ::-1], img_b[:, :, ::-1], img_r[:, :, ::-1]])):
            break


def test_iiw_dataset():
    from configs.intrinsic_iiw import opt
    from utils import tensor2img, show_image
    use_norm = False
    dataset_iiw = DatasetIdIIW(opt.data, is_train=True, mode=0, batch_size=3)
    dataset = dataset_iiw.load()
    for i, data in enumerate(dataset):
        stacked_img = data['img_1']
        targets = data['target_1']
        # pdb.set_trace()
        # img_i, img_b, img_r = data['I'], data['B'], data['R']
        img_1 = tensor2img(stacked_img[0,:,:,:], mean=opt.data.image_mean, std=opt.data.image_std, use_norm=use_norm)
        img_2 = tensor2img(stacked_img[1,:,:,:], mean=opt.data.image_mean, std=opt.data.image_std, use_norm=use_norm)
        img_3 = tensor2img(stacked_img[2,:,:,:], mean=opt.data.image_mean, std=opt.data.image_std, use_norm=use_norm)
        if not show_image(cv2.hconcat([img_1[:, :, ::-1], img_2[:, :, ::-1], img_3[:, :, ::-1]])):
            break


if __name__ == '__main__':
    #test_iiw_dataset()
    test_id_dataset()
