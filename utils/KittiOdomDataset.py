import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import pykitti

BASE_DIR = r'C:\Users\Will Haley\Documents\GitHub\kitti_odometry\dataset'

class KittiOdomDataset():
    def __init__(self, sequences: range, batch_len: int):
        self.sequences_len = [
            4540,
            1100,
            4660,
            800,
            270,
            2760,
            1100,
            1100,
            4070,
            1590,
            1200
        ]
        self.dataset = []
        for seq in sequences:
            batch_datasets = []
            for batch_idx in range(0, self.sequences_len[seq], batch_len):
                seq_fname = f"0{seq}" if seq < 10 else str(seq)
                batch_end_frame = batch_idx+batch_len if batch_idx+batch_len <= self.sequences_len[seq] else self.sequences_len[seq]
                # print(f"----------- \nSeq: {seq} \n Batch frame range: {batch_idx}:{batch_end_frame-1}")

                batch = pykitti.odometry(BASE_DIR, seq_fname, frames=(batch_idx, batch_end_frame, 1))

                # Adjust the poses so that the first frame in the batch is the global origin.
                init_pose = batch.poses[0]
                assert np.linalg.det(init_pose) != 0, "Pose matrix is singular! This shouldn't ever be the case AFAIK..."
                init_pose_inverse = np.linalg.inv(init_pose)
                batch.poses = [np.matmul(init_pose_inverse, pose) for pose in batch.poses]
                # batch.get_cam2()
                # batch.poses
                batch_datasets.append(batch)
            self.dataset.append(batch_datasets)






class _KittiOdomDataset(torch.utils.data.Dataset):
    def _maybe_resize_image(self, image, size):
        # size is (width, height)
        width, height = image.size
        if width != size[0] or height != size[1]:
            image = image.resize(size)
        return image

    def __init__(self, sequence):
        if sequence < 10:
            sequence = f"0{sequence}"
        else:
            sequence = str(sequence)
        fpath = f'datasets/data_odometry_csv/{sequence}.csv'
        dataframe = pd.read_csv(fpath)
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
    
        return img_tensor, odom_pose

def get_dataloaders(sequences, batch_size):
    datasets = []
    if isinstance(sequences, list):
        for sequence in sequences:
            datasets.append(KittiOdomDataset(sequence))
    elif isinstance(sequences, tuple):
        for sequence in range(*sequences):
            datasets.append(KittiOdomDataset(sequence))
    else:
        raise Exception("Sequences is not list or tuple!")
    
    return [DataLoader(dataset, batch_size=batch_size, pin_memory=True) for dataset in datasets]


# Code to verify dataloader is working
if __name__ == '__main__':
    dataset = KittiOdomDataset(range(0,8), 100)

    # Plot sequence 0, with adjusted poses
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for batch_dataset in dataset.dataset[0]:
    #     ground_truth = np.array(batch_dataset.poses)
    #     ax.plot(ground_truth[:,:,3][:,0], ground_truth[:,:,3][:,1], ground_truth[:,:,3][:,2])

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()
