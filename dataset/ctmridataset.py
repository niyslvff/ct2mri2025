import random
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import h5py
import sys
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append("..")
from utils import showSlices
from torch.utils.data import DataLoader



def random_position_crop3d(ct_img, mr_img=None, crop_rate=(0.5, 0.5, 0.5)):
    """
    create time: 2023/9/18
    modify content:
    1_v7_2023_9_25. correct bug. the random positions of ct img and mr img are different.
    2_v8_2023_9_25. correct bug. change crop_size to crop_rate for using different size of test image.
    """
    z, x, y = ct_img.shape
    crop_z, crop_x, crop_y = int(z * crop_rate[0]), int(x * crop_rate[1]), int(y * crop_rate[2])

    rand_z_start = np.random.randint(0, z - crop_z)
    rand_x_start = np.random.randint(0, x - crop_x)
    rand_y_start = np.random.randint(0, y - crop_y)
    new_ct_img = ct_img[rand_z_start:rand_z_start + crop_z, \
                 rand_x_start:rand_x_start + crop_x, \
                 rand_y_start:rand_y_start + crop_y]

    new_mr_img = mr_img[rand_z_start:rand_z_start + crop_z, \
                    rand_x_start:rand_x_start + crop_x, \
                    rand_y_start:rand_y_start + crop_y]

    return new_ct_img, new_mr_img


def random_position_crop(ct_img, mr_img=None, crop_rate=0.5):
    shape_tuple = ct_img.shape
    crop_shape_list = [int(shape_tuple[i] * crop_rate) for i in range(len(shape_tuple))]
    rand_shape_start_list = [np.random.randint(0, shape_tuple[i] - crop_shape_list[i]) for i in range(len(shape_tuple))]
    new_mr_img = None
    new_ct_img = None
    if len(shape_tuple) == 3:
        new_ct_img = ct_img[rand_shape_start_list[0]:rand_shape_start_list[0] + crop_shape_list[0],
                     rand_shape_start_list[1]:rand_shape_start_list[1] + crop_shape_list[1],
                     rand_shape_start_list[2]:rand_shape_start_list[2] + crop_shape_list[2]]
        if mr_img is not None:
            new_mr_img = mr_img[rand_shape_start_list[0]:rand_shape_start_list[0] + crop_shape_list[0],
                         rand_shape_start_list[1]:rand_shape_start_list[1] + crop_shape_list[1],
                         rand_shape_start_list[2]:rand_shape_start_list[2] + crop_shape_list[2]]
    if len(shape_tuple) == 2:
        new_ct_img = ct_img[rand_shape_start_list[0]:rand_shape_start_list[0] + crop_shape_list[0],
                     rand_shape_start_list[1]:rand_shape_start_list[1] + crop_shape_list[1]]

        if mr_img is not None:
            new_mr_img = mr_img[rand_shape_start_list[0]:rand_shape_start_list[0] + crop_shape_list[0],
                         rand_shape_start_list[1]:rand_shape_start_list[1] + crop_shape_list[1]]

    return new_ct_img, new_mr_img


class Resize(object):
    def __init__(self, output_size, mode='nearest'):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        batch = sample.unsqueeze(0).unsqueeze(0)
        batch = torch.nn.functional.interpolate(batch, size=self.output_size, mode=self.mode)
        sample = batch.squeeze(0).squeeze(0)
        return sample


def five_crop_3d(img, position):
    z, x, y = img.shape
    x, y = x // 2, y // 2
    transform = transforms.FiveCrop((x, y))
    new_tensor = torch.zeros((z, x, y))
    for i in range(z):
        new_tensor[i] = transform(img[i])[position]
    return new_tensor


def get_sorted_files(img_root_path, img_type):
    names = [
        os.path.join(img_root_path, img_type, name)
        for name in os.listdir(os.path.join(img_root_path, img_type))
    ]
    names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return names


class Raw3dDiffusionDataset(Dataset):
    r"""modified on 2024.12.30."""
    
    def __init__(self, img_num, img_root_path='crop\\brain', img_shape=(256, 256, 256),
                 crop_rate=(0.5, 0.5, 0.5), use_crop=True, is_val_set=False, transform=None):
        super(Raw3dDiffusionDataset, self).__init__()
        ct_img_names = get_sorted_files(img_root_path, 'ct')
        mr_img_names = get_sorted_files(img_root_path, 'mr')

        # shuffle index
        total_length = len(ct_img_names)
        self.use_length = total_length * 70 // 100 if not is_val_set else \
            total_length - total_length * 70 // 100
        # self.use_length = total_length
        shuffle_index = list(range(0, self.use_length)) \
            if not is_val_set else list(range(total_length - self.use_length, total_length))
        self.ct_img_names = [ct_img_names[index] for index in shuffle_index]
        self.mr_img_names = [mr_img_names[index] for index in shuffle_index]

        self.img_shape = img_shape
        if not transform:
            transform = torch.from_numpy
        self.transform = transform
        self.crop_rate = crop_rate
        self.img_num = img_num or total_length
        self.use_crop = use_crop

    def __getitem__(self, index):
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        ct_tensor = self.transform(
            np.reshape(np.fromfile(self.ct_img_names[index % self.use_length], dtype='float32'), self.img_shape))
        torch.random.manual_seed(seed)
        mr_tensor = self.transform(
            np.reshape(np.fromfile(self.mr_img_names[index % self.use_length], dtype='float32'), self.img_shape))

        if self.use_crop:
            ct_tensor, mr_tensor = random_position_crop3d(ct_tensor, mr_tensor, self.crop_rate)

        ct_tensor = ct_tensor.unsqueeze(0)
        mr_tensor = mr_tensor.unsqueeze(0)

        return ct_tensor, mr_tensor

    def __len__(self):
        return self.img_num


class Raw3dDiffusionMajiDataset(Dataset):
    r"""modified on 2024.12.30."""
    def __init__(
            self,
            img_num=1,
            img_root_path='./crop/preprocess_globalNormAndEnContrast',
            img_shape=(160, 224, 168),
            is_val=False,
            use_crop=False,
            crop_rate=(0.5, 0.5, 0.5),
            crop_size=None,
            transform=None,
            is_used_to_concate=False
    ):
        super(Raw3dDiffusionMajiDataset, self).__init__()
        ct_img_names = [
            os.path.join(img_root_path, name)
            for name in os.listdir(img_root_path) if 'ct' in name
        ]
        ct_img_names.sort(key=lambda x: int(x.split('.')[3]))

        mr_img_names = [
            os.path.join(img_root_path, name)
            for name in os.listdir(img_root_path) if 'mr' in name
        ]
        mr_img_names.sort(key=lambda x: int(x.split('.')[3]))

        total_length = len(ct_img_names)
        train_length = total_length * 70 // 100
        val_length = total_length - total_length * 70 // 100
        train_index = list(range(0, train_length))
        val_index = list(range(total_length - val_length, total_length))

        train_ct_img_names = [ct_img_names[index] for index in train_index]
        val_ct_img_names = [ct_img_names[index] for index in val_index]

        train_mr_img_names = [mr_img_names[index] for index in train_index]
        val_mr_img_names = [mr_img_names[index] for index in val_index]
        if img_num >= 1 and is_used_to_concate and not is_val:
            self.ct_img_names = []
            self.mr_img_names = []
            for i in range(img_num):
                self.ct_img_names += train_ct_img_names
                self.mr_img_names += train_mr_img_names
            for i in range(img_num):
                self.ct_img_names += val_ct_img_names
                self.mr_img_names += val_mr_img_names
            self.use_length = img_num * (train_length + val_length)
        else:
            self.ct_img_names = train_ct_img_names if not is_val else val_ct_img_names
            self.mr_img_names = train_mr_img_names if not is_val else val_mr_img_names
            self.use_length = train_length if not is_val else val_length

        self.img_shape = img_shape
        if not transform:
            transform = torch.from_numpy
        self.transform = transform
        self.use_crop = use_crop
        self.crop_rate = crop_rate
        self.crop_size = crop_size

    def __getitem__(self, index):
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        ct_tensor = self.transform(
            np.reshape(np.fromfile(self.ct_img_names[index % self.use_length], dtype='float32'), self.img_shape))
        torch.random.manual_seed(seed)
        mr_tensor = self.transform(
            np.reshape(np.fromfile(self.mr_img_names[index % self.use_length], dtype='float32'), self.img_shape))

        if self.use_crop:
            ct_tensor, mr_tensor = random_position_crop3d(ct_tensor, mr_tensor, crop_rate=self.crop_rate)

        ct_tensor = ct_tensor.unsqueeze(0)
        mr_tensor = mr_tensor.unsqueeze(0)

        return ct_tensor, mr_tensor

    def __len__(self):
        return self.use_length


class Raw3dDiffusionConcatenateDataset(Dataset):
    r"""modified on 2024.12.30."""
    def __init__(self, img_num, img_root_path='crop\\brain', img_shape=(160, 224, 160),
                 crop_rate=(0.5, 0.5, 0.5), use_crop=True, is_val_set=False, transform=None):
        super(Raw3dDiffusionConcatenateDataset, self).__init__()
        mr_img_names = [
            os.path.join(img_root_path, 'mri_SR3UNet_train_cond_2D_v0', name)
            for name in os.listdir(os.path.join(img_root_path, 'mri_SR3UNet_train_cond_2D_v0'))
        ]
        try:
            mr_img_names.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        except:
            mr_img_names.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        concatenate_mr_img_names = [
            os.path.join(img_root_path, 'pred_mri_SR3UNet_train_cond_2D_v0', name)
            for name in os.listdir(os.path.join(img_root_path, 'pred_mri_SR3UNet_train_cond_2D_v0'))
        ]
        try:
            concatenate_mr_img_names.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        except:
            concatenate_mr_img_names.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # shuffle index
        total_length = len(concatenate_mr_img_names)
        self.use_length = total_length * 70 // 100 if not is_val_set else \
            total_length - total_length * 70 // 100
        # self.use_length = total_length
        shuffle_index = list(range(0, self.use_length)) \
            if not is_val_set else list(range(total_length - self.use_length, total_length))
        self.concatenate_mr_img_names = [concatenate_mr_img_names[index] for index in shuffle_index]
        self.mr_img_names = [mr_img_names[index] for index in shuffle_index]

        self.img_shape = img_shape
        if not transform:
            transform = torch.from_numpy
        self.transform = transform
        self.crop_rate = crop_rate
        self.img_num = self.use_length
        self.use_crop = use_crop

    def __getitem__(self, index):
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        concatenate_mr_tensor = self.transform(
            np.reshape(np.fromfile(self.concatenate_mr_img_names[index % self.use_length], dtype='float32'),
                       self.img_shape))
        torch.random.manual_seed(seed)
        mr_tensor = self.transform(
            np.reshape(np.fromfile(self.mr_img_names[index % self.use_length], dtype='float32'), self.img_shape))

        if self.use_crop:
            concatenate_mr_tensor, mr_tensor = random_position_crop3d(concatenate_mr_tensor, mr_tensor, self.crop_rate)

        concatenate_mr_tensor = concatenate_mr_tensor.unsqueeze(0)
        mr_tensor = mr_tensor.unsqueeze(0)

        return concatenate_mr_tensor, mr_tensor

    def __len__(self):
        return self.img_num


class Raw3dDiffusionConcatenateAndCTDataset(Dataset):
    def __init__(self, img_num, img_root_path='crop\\brain', img_shape=(160, 224, 160),
                 crop_rate=(0.5, 0.5, 0.5), use_crop=True, is_val_set=False, transform=None):
        super(Raw3dDiffusionConcatenateAndCTDataset, self).__init__()
        mr_img_names = [
            os.path.join(img_root_path, 'mri_SR3UNet_train_cond_2D_v0', name)
            for name in os.listdir(os.path.join(img_root_path, 'mri_SR3UNet_train_cond_2D_v0'))
        ]
        try:
            mr_img_names.sort(key=lambda x: int(x.split('_')[7].split('\\')[1].split('.')[0]))
        except IndexError:
            mr_img_names.sort(key=lambda x: int(x.split('_')[7].split('/')[1].split('.')[0]))

        concatenate_mr_img_names = [
            os.path.join(img_root_path, 'pred_mri_SR3UNet_train_cond_2D_v0', name)
            for name in os.listdir(os.path.join(img_root_path, 'pred_mri_SR3UNet_train_cond_2D_v0'))
        ]
        try:
            concatenate_mr_img_names.sort(key=lambda x: int(x.split('_')[8].split('\\')[1].split('.')[0]))
        except IndexError:
            concatenate_mr_img_names.sort(key=lambda x: int(x.split('_')[8].split('/')[1].split('.')[0]))

        ct_img_names = [
            os.path.join('../crop/preprocess_globalNormAndEnContrast', name)
            for name in os.listdir('../crop/preprocess_globalNormAndEnContrast') if 'ct' in name
        ]
        ct_img_names.sort(key=lambda x: int(x.split('_')[3].split('.')[1]))
        ct_img_names = ct_img_names[:70] + ct_img_names[:70] + ct_img_names[70:] + ct_img_names[70:]

        # shuffle index
        total_length = len(concatenate_mr_img_names)
        self.use_length = total_length * 70 // 100 if not is_val_set else \
            total_length - total_length * 70 // 100
        # self.use_length = total_length
        shuffle_index = list(range(0, self.use_length)) \
            if not is_val_set else list(range(total_length - self.use_length, total_length))
        self.concatenate_mr_img_names = [concatenate_mr_img_names[index] for index in shuffle_index]
        self.mr_img_names = [mr_img_names[index] for index in shuffle_index]
        self.ct_img_names = [ct_img_names[index] for index in shuffle_index]

        self.img_shape = img_shape
        if not transform:
            transform = torch.from_numpy
        self.transform = transform
        self.crop_rate = crop_rate
        self.img_num = img_num or self.use_length
        self.use_crop = use_crop

    def __getitem__(self, index):
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        concatenate_mr_tensor = self.transform(
            np.reshape(np.fromfile(self.concatenate_mr_img_names[index % self.use_length], dtype='float32'),
                       self.img_shape))
        torch.random.manual_seed(seed)
        mr_tensor = self.transform(
            np.reshape(np.fromfile(self.mr_img_names[index % self.use_length], dtype='float32'), self.img_shape))
        torch.random.manual_seed(seed)
        ct_tensor = self.transform(
            np.reshape(np.fromfile(self.ct_img_names[index % self.use_length], dtype='float32'), self.img_shape)
        )

        if self.use_crop:
            concatenate_mr_tensor, mr_tensor, ct_tensor = random_position_crop3d(
                concatenate_mr_tensor, mr_tensor, ct_tensor, self.crop_rate
            )

        concatenate_mr_tensor = concatenate_mr_tensor.unsqueeze(0)
        mr_tensor = mr_tensor.unsqueeze(0)
        ct_tensor = ct_tensor.unsqueeze(0)

        return concatenate_mr_tensor, mr_tensor, ct_tensor

    def __len__(self):
        return self.img_num


class Raw3dMixedDiffusionDataset(Dataset):
    def __init__(self, img_type='mixed',
                 img_root_path='crop\\brain2', img_shape=(160, 224, 160),
                 use_crop=False, crop_rate=(0.5, 0.5, 0.5),
                 is_val_set=False, transform=None):
        super(Raw3dMixedDiffusionDataset, self).__init__()
        if img_shape[2] != 168 and img_shape[0] != 96:
            ct_img_names = get_sorted_files(img_root_path, 'ct')
            mr_img_names = get_sorted_files(img_root_path, 'mr')
        else:
            ct_img_names = [
                os.path.join(img_root_path, name)
                for name in os.listdir(img_root_path) if 'ct' in name
            ]
            ct_img_names.sort(key=lambda x: int(x.split('.')[3]))

            mr_img_names = [
                os.path.join(img_root_path, name)
                for name in os.listdir(img_root_path) if 'mr' in name
            ]
            mr_img_names.sort(key=lambda x: int(x.split('.')[3]))

        total_length = len(ct_img_names)
        self.use_length = total_length * 70 // 100 if not is_val_set else \
            total_length - total_length * 70 // 100

        start = 0 if not is_val_set else self.use_length
        end = self.use_length if not is_val_set else -1

        if img_type == "mixed":
            names = ct_img_names[start:end] + mr_img_names[start:end]
        elif img_type == "ct":
            names = ct_img_names[start:end]
        elif img_type == "mr":
            names = mr_img_names[start:end]
        else:
            raise "wrong image type"
        self.names = names

        if not transform:
            transform = torch.from_numpy
        self.transform = transform
        self.img_shape = img_shape
        self.use_crop = use_crop
        self.crop_rate = crop_rate

        self.real_length = len(self.names)
        self.is_val = is_val_set
        self.img_type = img_type

    def __getitem__(self, index):
        if self.use_crop:
            imgs_tensor = self.transform(random_position_crop(np.fromfile(self.names[index % self.real_length], dtype=np.float32).reshape(self.img_shape))[0])
        else:
            imgs_tensor = self.transform(
                np.fromfile(self.names[index % self.real_length], dtype=np.float32).reshape(self.img_shape))

        return torch.unsqueeze(imgs_tensor, dim=0), 1

    def __len__(self):
        return self.real_length


class Diff2dDataset(Dataset):
    def __init__(self, file_path, crop=False, crop_scale=0.5, num=540, is_val=False):
        sel_length = int(540 * 0.8)
        with h5py.File(file_path, "r") as f:
            self.ct_imgs = f['ct'][:sel_length] if not is_val else f['ct'][sel_length:-1]
            self.mr_imgs = f['mr'][:sel_length] if not is_val else f['mr'][sel_length:-1]

        self.real_length = len(self.ct_imgs)
        self.crop = crop
        self.crop_scale = crop_scale
        self.num = num
        self.is_val = is_val
        # if crop:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         # TODO: RandomCrop
        #         # transforms.CenterCrop(self.ct_imgs[0].shape[0] * crop_scale),
        #         transforms.Resize((64, 64), antialias=True),
        #     ])
        # else:
        self.transform = torch.from_numpy

    def __getitem__(self, index):
        """

        :param index:
        :return: an [1 x W x H] Tensor
        """
        ct = self.ct_imgs[index % self.real_length]
        mr = self.mr_imgs[index % self.real_length]
        if self.crop:
            ct, mr = random_position_crop(ct, mr, self.crop_scale)
        ct_tensor = self.transform(ct)
        mr_tensor = self.transform(mr)
        return ct_tensor, mr_tensor

    def __len__(self):
        return self.num


class MixedSize2DDataset(Dataset):
    def __init__(self, file_path="../MIXED_2D_IMGS_GP(MIXED_SIZE).hdf5", crop=False, is_val=False, my_transform=None,
                 img_num=None):
        with h5py.File(file_path, "r") as f:
            imgs_160224 = f['imgs_160224'][:]
            imgs_160168 = f['imgs_160168'][:]
            imgs_224168 = f['imgs_224168'][:]

            img_160224_labels = f['img_160224_labels'][:]
            img_160168_labels = f['img_160168_labels'][:]
            img_224168_labels = f['img_224168_labels'][:]

        # assert len(imgs_160168) == len(imgs_160224) and len(imgs_160224) == len(imgs_224168), "inconsistent length."
        print(imgs_160168[0].shape)
        assert imgs_160168[0].shape == imgs_160224[0].shape and imgs_160168[0].shape == imgs_224168[0].shape
        print(f"amount of 160224: {len(imgs_160224)}, amount of 160168: {len(imgs_160168)}, amount of 224168: {len(imgs_224168)}")
        min_list_length = min(len(imgs_160168), len(imgs_160224), len(imgs_224168))
        
        # imgs_224168 = imgs_224168[:min_list_length, :, :]
        # imgs_160168 = imgs_160168[:min_list_length, :, :]
        # imgs_160224 = imgs_160224[:min_list_length, :, :]
        # img_224168_labels = img_224168_labels[:min_list_length]
        # img_160168_labels = img_160168_labels[:min_list_length]
        # img_160224_labels = img_160224_labels[:min_list_length]
        
        ct_imgs, mr_imgs = [], []


        for i in range(min_list_length):
            if img_160224_labels[i] == 0:
                ct_imgs.append(imgs_160224[i])
                ct_imgs.append(imgs_160168[i])
                ct_imgs.append(imgs_224168[i])
            else:
                mr_imgs.append(imgs_160224[i])
                mr_imgs.append(imgs_160168[i])
                mr_imgs.append(imgs_224168[i])

        self.real_length = int(len(mr_imgs) * 0.7) if not is_val else (len(mr_imgs) - int(len(mr_imgs) * 0.7))
        start_index = 0 if not is_val else int(len(mr_imgs) * 0.7)
        end_index = int(len(mr_imgs) * 0.7) if not is_val else len(mr_imgs)
        self.ct_imgs = ct_imgs[start_index:end_index]
        self.mr_imgs = mr_imgs[start_index:end_index]
        self.crop = crop

        if my_transform is not None:
            self.transform = my_transform
        else:
            self.transform = torch.from_numpy

    def __getitem__(self, index):
        seed = torch.random.seed()
        if self.crop:
            cts, mris = random_position_crop(
                self.ct_imgs[index % self.real_length],
                self.mr_imgs[index % self.real_length],
                crop_rate=0.25,
            )
        else:
            cts = self.ct_imgs[index % self.real_length]
            mris = self.mr_imgs[index % self.real_length]
        torch.random.manual_seed(seed)
        ct_tensors = self.transform(cts)
        torch.random.manual_seed(seed)
        mr_tensors = self.transform(mris)

        return ct_tensors.unsqueeze(0), mr_tensors.unsqueeze(0)

    def __len__(self):
        return self.real_length


if __name__ == "__main__":
    # train_dataset = Raw3dDataset(200)
    # item_pair = train_dataset.__getitem__(190)
    # print(item_pair[0].shape)

    # dataset = Diff2dDataset("2DCTMRISET.hdf5", crop=True)
    # item_pair = dataset.__getitem__(11)
    # plt.subplot(1, 2, 1)
    # print(item_pair[0][0].shape)
    # plt.imshow(item_pair[0][0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(item_pair[1][0])
    #
    # plt.show()

    transform = transforms.Compose([
        torch.from_numpy,
        # Resize((128, 128, 128), mode='trilinear'),
        transforms.RandomHorizontalFlip(0.5),
    ])

    # dataset = Diff2dMixedImgWithLabelDataset(
    #     "MIXED_IMGS_WITH_LABELS2.hdf5",
    #     img_type='pair',
    #     crop=True,
    #     my_transform=transform
    # )
    # print(dataset.__len__(), len(dataset.ct_imgs), len(dataset))
    # slice_list = []
    # for i in range(1):
    #     item_pair = dataset.__getitem__(i)
    #     print(item_pair[0].shape)
    #     slice_list.append(item_pair[0][0])
    #     slice_list.append(item_pair[1][0])
    # showSlices(slice_list, rows=1)

    # dataset = Raw3dDiffusionConcatenateDataset(30,
    #                                            img_root_path='./crop/concatenate_mri_pairs',
    #                                            img_shape=(160, 224, 160),
    #                                            use_crop=True, crop_rate=(0.5, 0.5, 0.5), is_val_set=False,
    #                                            transform=transform)
    # item_pair = dataset.__getitem__(0)
    # ct, mr = item_pair[0][0], item_pair[1][0]
    # slice_list = []
    # for item in [ct, mr]:
    #     slice_list.append(item[item.shape[0] // 2, :, :])
    #     slice_list.append(item[:, item.shape[1] // 2, :])
    #     slice_list.append(item[:, :, item.shape[2] // 2])
    # showSlices(slice_list, rows=2)
    # print(torch.max(ct), torch.min(ct))
    # print(torch.max(mr), torch.min(mr))

    # dataset = Raw3dMixedDiffusionDataset('mr', use_crop=True)
    # items = []
    # for i in range(5):
    #     items.append(dataset.__getitem__(i))
    #
    # slice_list = []
    # for item, _ in items:
    #     print(item.shape)
    #     slice_list.append(item[0][item[0].shape[0] // 2, :, :])
    #     slice_list.append(item[0][:, item[0].shape[1] // 2, :])
    #     slice_list.append(item[0][:, :, item[0].shape[2] // 2])
    # showSlices(slice_list, rows=5)

    # dataset = Raw3dDiffusion160224168Dataset(
    #     30,
    #     img_root_path='./crop/preprocess_globalNormAndEnContrast',
    #     img_shape=(160, 224, 168),
    #     use_crop=True, crop_rate=(0.5, 0.5, 0.5), is_val_set=False,
    #     transform=transform
    # )
    # item_pair = dataset.__getitem__(0)
    # ct, mr = item_pair[0][0], item_pair[1][0]
    # slice_list = []
    # for item in [ct, mr]:
    #     slice_list.append(item[item.shape[0] // 2, :, :])
    #     slice_list.append(item[:, item.shape[1] // 2, :])
    #     slice_list.append(item[:, :, item.shape[2] // 2])
    # showSlices(slice_list, rows=2)

    # with h5py.File("MIXED_2D_IMGS(MIXED_SIZE).hdf5", "r") as f:
    #     imgs_160224 = f['imgs_160224'][:]
    #     imgs_160168 = f['imgs_160168'][:]
    #     imgs_224168 = f['imgs_224168'][:]
    #
    # slice_list = []
    # for i in range(0, 4, 2):
    #     slice_list.append(imgs_160224[i])
    #     slice_list.append(imgs_224168[i])
    #     slice_list.append(imgs_160168[i])
    # showSlices(slice_list, rows=3)

    dataset = MixedSize2DDataset(crop=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        ct, mri = batch
        print(ct.shape)
        slices = [ct.squeeze().squeeze(), mri.squeeze().squeeze()]
        showSlices(slices, 1)
        # break
    


    # def collate_fn(batch):
    #     # 检查每个元组中的第一个 tensor 的形状是否与 batch 中的第一个元组的第一个 tensor 的形状相同
    #     sel_size = random.randint(0, 2)
    #     print(sel_size, batch[sel_size][0].shape)
    #     batch = [data for data in batch if data[0].shape == batch[sel_size][0].shape]

    #     # 将每个元组中的 tensor 拆解并合并成一个 tensor
    #     data_tensor = torch.stack([data[0] for data in batch])
    #     label_tensor = torch.stack([data[1] for data in batch])

    #     return data_tensor, label_tensor


    # class CustomDataLoader(DataLoader):
    #     def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
    #                  batch_sampler=None, num_workers=0, collate_fn=None,
    #                  pin_memory=False, drop_last=False, timeout=0,
    #                  worker_init_fn=None, multiprocessing_context=None):
    #         super(CustomDataLoader, self).__init__(
    #             dataset, batch_size, shuffle, sampler,
    #             batch_sampler, num_workers, collate_fn,
    #             pin_memory, drop_last, timeout,
    #             worker_init_fn, multiprocessing_context)

    #     def __iter__(self):
    #         fix_ct, fix_mr = next(super(CustomDataLoader, self).__iter__())
    #         fix_shape = fix_ct.shape[2:]
    #         print(fix_ct.shape[0])
    #         for ct, mr in super(CustomDataLoader, self).__iter__():
    #             # 获取 batch 中每个 item 的尺寸
    #             if ct.shape[2:] == fix_shape:
    #                 # 如果尺寸不一致，重复添加 item 直到达到 batch_size
    #                 if ct.shape[0] + fix_ct.shape[0] == self.batch_size:
    #                     empty_ct = np.zeros((self.batch_size, 1, *fix_shape), dtype="float32")
    #                     empty_ct[:fix_ct.shape[0], :, :, :] = fix_ct
    #                     empty_ct[fix_ct.shape[0]:, :, :, :] = ct
    #                     fix_ct = torch.from_numpy(empty_ct)
    #                     empty_mr = np.zeros((self.batch_size, 1, *fix_shape), dtype="float32")
    #                     empty_mr[:fix_mr.shape[0], :, :, :] = fix_mr
    #                     empty_mr[fix_mr.shape[0]:, :, :, :] = mr
    #                     fix_mr = torch.from_numpy(empty_mr)
    #                     yield fix_ct, fix_mr
    #                 elif ct.shape[0] + fix_ct.shape[0] < self.batch_size:
    #                     small_size = ct.shape[0] + fix_ct.shape[0]
    #                     empty_ct = np.zeros((small_size, 1, *fix_shape), dtype="float32")
    #                     empty_ct[:fix_ct.shape[0], :, :, :] = fix_ct
    #                     empty_ct[fix_ct.shape[0]:, :, :, :] = ct
    #                     fix_ct = torch.from_numpy(empty_ct)
    #                     empty_mr = np.zeros((small_size, 1, *fix_shape), dtype="float32")
    #                     empty_mr[:fix_mr.shape[0], :, :, :] = fix_mr
    #                     empty_mr[fix_mr.shape[0]:, :, :, :] = mr
    #                     fix_mr = torch.from_numpy(empty_mr)
    #                 else:
    #                     ct = ct[:self.batch_size - fix_ct.shape[0], :, :, :]
    #                     mr = mr[:self.batch_size - fix_mr.shape[0], :, :, :]
    #                     empty_ct = np.zeros((self.batch_size, 1, *fix_shape), dtype="float32")
    #                     empty_ct[:fix_ct.shape[0], :, :, :] = fix_ct
    #                     empty_ct[fix_ct.shape[0]:, :, :, :] = ct
    #                     fix_ct = torch.from_numpy(empty_ct)
    #                     empty_mr = np.zeros((self.batch_size, 1, *fix_shape), dtype="float32")
    #                     empty_mr[:fix_mr.shape[0], :, :, :] = fix_mr
    #                     empty_mr[fix_mr.shape[0]:, :, :, :] = mr
    #                     fix_mr = torch.from_numpy(empty_mr)
    #                     yield fix_ct, fix_mr


    # # dataloader = DataLoader(dataset=dataset, batch_size=35, drop_last=True,
    # #                         # shuffle=True,
    # #                         num_workers=0,
    # #                         collate_fn=collate_fn
    # #                         )
    # dataloader = CustomDataLoader(dataset=dataset, batch_size=35, drop_last=True, shuffle=True, collate_fn=collate_fn)
    # print(dataset.__len__())
    # slices = []
    # sel_dh, sel_dw, sel_hw = list(range(40)), list(range(40)), list(range(40))
    # for idx, img in enumerate(dataloader):
    #     print(len(img[0]))
    #     slices.append(img[0][0][0])
    #     slices.append(img[1][0][0])
    #     if idx == 5:
    #         break
    # showSlices(slices, rows=3)

    # plt.show()
