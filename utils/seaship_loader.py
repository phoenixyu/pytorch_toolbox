import sys
import torch
import torchvision
import numpy as np
import cv2
import random
import collections

import matplotlib.pyplot as plt
import scipy.misc as m
from tqdm import tqdm
from torch.utils import data
from skimage.transform import rotate
from skimage.transform import resize

class SeashipLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=256, n_classes=5):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = n_classes
        self.img_size = img_size if isinstance(img_size, tuple) else(img_size, img_size)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.files = collections.defaultdict(list)
    
        for split in ["train", "val", "trainval"]:
            file_list = tuple(open(root + '/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = "{}/img/{}".format(self.root, img_name)
        lbl_path = "{}/label/{}".format(self.root, img_name)
        
        img = cv2.imread(img_path).astype(np.uint8)
        lbl = cv2.imread(lbl_path).astype(np.int32)
        lbl = lbl[:,:,1]
        lbl = lbl.reshape((lbl.shape[0], lbl.shape[1]))

        # 数据增强
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        return img, lbl
    
    def transform(self, img, lbl):
        """
        截取图像部分区域=>旋转水平翻转=>垂直翻转=>变换到所需图片尺寸
        """
        img = img[:, :, ::-1].astype(np.float64)
        lbl[lbl==255] = 0 

        # random scale size crop
        use_randomcrop = True
        if use_randomcrop:
            if self.split != 'val':
                for attempt in range(100):
                    areas = img.shape[0] * img.shape[1]
                    target_area = random.uniform(0.5, 1) * areas
                    
                    w, h = int(round(np.sqrt(target_area))), int(round(np.sqrt(target_area)))

                    if w <= img.shape[1] and h <= img.shape[0]:
                        x1 = random.randint(0, img.shape[1] - w)
                        y1 = random.randint(0, img.shape[0] - h)

                        img = img[y1:y1+h, x1:x1+w]
                        lbl = lbl[y1:y1+h, x1:x1+w]
                        if(((img.shape[1],img.shape[0]) == (w, h)) and ((lbl.shape[1],lbl.shape[0]) == (w, h))):
                            break
                assert((img.shape[1],img.shape[0]) == (w, h))
                assert((lbl.shape[1],lbl.shape[0]) == (w, h))
        else:
            w, h = img.shape[1], img.shape[0]
            new_w, new_h = self.img_size[1], self.img_size[0]
            x1 = int(round((w - new_w) / 2.))
            y1 = int(round((h - new_h) / 2.))
            img = img[y1:y1+h,x1:x1+w]
            lbl = lbl[y1:y1+h,x1:x1+w]

        # random rotate
        if random.random() < 0.5 and self.split != 'val':
            angle = random.randint(-90, 90)
            img = rotate(img, angle=angle, mode='symmetric', preserve_range=True)
            lbl = rotate(lbl, angle=angle, mode='symmetric', order=1, preserve_range=True)
        lbl = lbl.astype(np.int32)

        # print(np.unique(lbl[:,:,0]==lbl[:,:,2]))
        # random vert flip
        if random.random() < 0.5 and self.split != 'val':
            img = np.flip(img, axis=0)
            lbl = np.flip(lbl, axis=0)
        
        # random hor flip
        if random.random() < 0.5 and self.split != 'val':
            img = np.flip(img, axis=1)
            lbl = np.flip(lbl, axis=1)
        
        img = resize(img, (self.img_size[0], self.img_size[1]), mode='symmetric', preserve_range=True)
        img = img.astype(float) / 255.0
        img = img - self.mean
        img = img / self.std

        lbl = lbl.astype(float)
        lbl = resize(lbl, (self.img_size[0], self.img_size[1]), mode='symmetric', preserve_range=True)
        lbl = lbl.astype(int)
        
        # NHWC => NCHW
        img = img.transpose(2,0,1)
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        
        return img, lbl

    def get_labels(self):
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128]])

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/dl/phoenix_lzx/torch/data/dataset/seaship-train'
    dst = SeashipLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=1,shuffle=False)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        # img = torchvision.utils.make_grid(imgs).numpy()
        # img = np.transpose(img, (1, 2, 0))
        # img = img[:, :, ::-1]
        # plt.figure(1)
        # plt.imshow(img)
        # plt.show(block=False)

        # plt.figure(2)
        # plt.imshow(dst.decode_segmap(labels.numpy()[0]))
        # plt.show()