import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
from PIL import Image
import os

patch_size, stride = 40, 30
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, data_dir, label_dir, patch_size=40, stride=30,
                 scales=(1, 0.9, 0.8, 0.7), aug_times=1):
        super(DenoisingDataset, self).__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.scales = scales

        # Build patch index: (filename, scale, row, col)
        # Augmentation is randomized in __getitem__ for better diversity
        self.patches = []
        file_list = sorted(os.listdir(data_dir))
        for fname in file_list:
            img = np.array(Image.open(os.path.join(data_dir, fname)))
            if img.ndim == 3:
                img = img[:, :, 0]
            h, w = img.shape
            for s in scales:
                h_scaled, w_scaled = int(h * s), int(w * s)
                img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
                actual_h, actual_w = img_scaled.shape
                for i in range(0, actual_h - patch_size + 1, stride):
                    for j in range(0, actual_w - patch_size + 1, stride):
                        for _ in range(aug_times):
                            self.patches.append((fname, s, i, j))
        print(f'^_^-training data indexed: {len(self.patches)} patches-^_^')

    def __getitem__(self, index):
        fname, scale, row, col = self.patches[index]
        ps = self.patch_size

        img = np.array(Image.open(os.path.join(self.data_dir, fname)))
        label = np.array(Image.open(os.path.join(self.label_dir, fname)))
        if img.ndim == 3:
            img = img[:, :, 0]
        if label.ndim == 3:
            label = label[:, :, 0]

        h, w = img.shape
        h_scaled, w_scaled = int(h * scale), int(w * scale)
        img = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)

        x_patch = img[row:row + ps, col:col + ps]
        y_patch = label[row:row + ps, col:col + ps]

        aug_mode = np.random.randint(0, 8)
        x_patch = data_aug(x_patch, mode=aug_mode)
        y_patch = data_aug(y_patch, mode=aug_mode)

        x_patch = torch.from_numpy(x_patch.copy()).unsqueeze(0).float() / 255.0
        y_patch = torch.from_numpy(y_patch.copy()).unsqueeze(0).float() / 255.0

        return x_patch, y_patch

    def __len__(self):
        return len(self.patches)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name, data_dir, label_dir, mode):
    # get multiscale patches from a single image
    img = np.array(Image.open(os.path.join(data_dir, file_name)))
    img=img[:, :, 0]
    label = np.array(Image.open(os.path.join(label_dir, file_name)))
    label = label[:, :, 0]
    h, w = img.shape
    x_patches = []
    y_patches=[]
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        label_scaled = cv2.resize(label, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                y=label_scaled[i:i+patch_size, j:j+patch_size]
                start = 0
                # if mode == 'B':
                #     patches.append(x)
                #     start = 1
                for k in range(0, aug_times):
                    r_mode=np.random.randint(start, 8)
                    x_aug = data_aug(x, mode=r_mode)
                    y_aug=data_aug(y, mode=r_mode)
                    x_patches.append(x_aug)
                    y_patches.append(y_aug)
    return x_patches, y_patches


def datagenerator(batch_size, data_dir, label_dir, verbose=False, mode='S', channels=1):
    # generate clean patches from a dataset
    file_list = os.listdir(data_dir)  # get name list of all .png files
    # initrialize
    x_data = []
    y_data=[]
    # generate patches
    for i in range(len(file_list)):
        x_patches, y_patches = gen_patches(file_list[i], data_dir, label_dir, mode)
        for x_patch,y_patch in zip(x_patches, y_patches):
            x_data.append(x_patch)
            y_data.append(y_patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    x_data = np.array(x_data, dtype='uint8')
    x_data = np.expand_dims(x_data, axis=3)

    y_data = np.array(y_data, dtype='uint8')
    y_data = np.expand_dims(y_data, axis=3)
    # discard_n = len(x_data) - len(x_data) // batch_size * batch_size  # because of batch normalization
    # data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data prepared-^_^')
    return x_data, y_data


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    data = datagenerator(8,data_dir='E:/RDDCNN_train_data',label_dir='E:/RDDCNN_train_SPCT_data')  # 调用路径  修改
    data1 = np.array(data)
    print(data1.shape)
    # for d in data:     #循环训练2w多张 用服务器
    #for d in data[:10]:    #训练十张图片
        # show(d,'a')

   #
   # print('Shape of result = ' + str(res.shape))
   # print('Saving data...')
   # if not os.path.exists(save_dir):
   #         os.mkdir(save_dir)
   # np.save(save_dir+'clean_patches.npy', res)
   # print('Done.')