import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class KittiOdomDataset(torch.utils.data.Dataset):
    def _maybe_resize_image(self, image, size):
        # size is (width, height)
        width, height = image.size
        if width != size[0] or height != size[1]:
            image = image.resize(size)
        return image

    def __init__(self, csv_file):
        dataframe = pd.read_csv(csv_file)
        self.img_fpaths = dataframe["img_fpaths"].to_numpy()
        self.odom_poses = dataframe[[f"odom_{i}" for i in range(1, 13)]].to_numpy()

        self.received_invalid_image_sizes = set()

    def __len__(self):
        return len(self.img_fpaths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        image = Image.open(self.img_fpaths[index])
        image = self._maybe_resize_image(image, (1241, 376))
        img_tensor = transform(image)

        odom_pose = torch.from_numpy(self.odom_poses[index]).float()

        sample = {'image': img_tensor, 'odom': odom_pose}
        return sample
    

# Code to verify dataloader is working
if __name__ == '__main__':
    dataset = KittiOdomDataset(r'datasets\data_odometry_csv\00.csv')
    dataloader = DataLoader(dataset, batch_size=239)

    for batch_idx, sample in enumerate(dataloader):
        print(batch_idx, sample['image'].size(), sample['odom'].size())
        print(sample['odom'][0])
        transform = torchvision.transforms.ToPILImage()
        pil_img = transform(sample['image'][0])
        pil_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2RGB) 
        cv2.imshow('img',np.array(pil_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break