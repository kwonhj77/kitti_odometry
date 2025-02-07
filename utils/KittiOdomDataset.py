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


class KittiOdomBatch(torch.utils.data.Dataset):
    def _maybe_resize_image(self, image, size):
        # size is (width, height)
        width, height = image.size
        if width != size[0] or height != size[1]:
            image = image.resize(size)
        return image

    def __init__(self, sequence, frame_start_idx, frame_end_idx):
        seq_fname = f"0{sequence}" if sequence < 10 else str(sequence)
        # print(f"----------- \nSeq: {sequence} \n Batch frame range: {frame_start_idx}:{frame_end_idx-1}")

        batch = pykitti.odometry(BASE_DIR, seq_fname, frames=range(frame_start_idx, frame_end_idx, 1))

        # Adjust the poses so that the first frame in the batch is the global origin.
        init_pose = batch.poses[0]
        assert np.linalg.det(init_pose) != 0, "Pose matrix is singular! This shouldn't ever be the case AFAIK..."
        init_pose_inverse = np.linalg.inv(init_pose)
        batch.poses = [np.matmul(init_pose_inverse, pose) for pose in batch.poses]
        self.dataset = batch

    def __len__(self):
        return len(list(self.dataset.frames))
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        img = self.dataset.get_cam2(index)
        img = self._maybe_resize_image(img, (1241, 376))
        img_tensor = transform(img)
        pose = torch.from_numpy(self.dataset.poses[index].flatten()[:-4]).float()
    
        return img_tensor, pose

def _get_batches_from_sequence(sequence, sequence_len, batch_len):
    batch_datasets = []
    for frame_start_idx in range(0, sequence_len, batch_len):
        frame_end_idx = frame_start_idx+batch_len if frame_start_idx+batch_len <= sequence_len else sequence_len
        batch_datasets.append(KittiOdomBatch(sequence, frame_start_idx, frame_end_idx))
    return batch_datasets

def get_batches(sequences, batch_len):
    sequences_len = [
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
    batches = []
    if isinstance(sequences, tuple):
        sequences = range(*sequences)

    for sequence in sequences:
        batches.extend(_get_batches_from_sequence(sequence=sequence, sequence_len=sequences_len[sequence], batch_len=batch_len))

    return [DataLoader(batch, batch_size=len(batch), pin_memory=True) for batch in batches]


##########################################

# class _KittiOdomDataset(torch.utils.data.Dataset):
#     def _maybe_resize_image(self, image, size):
#         # size is (width, height)
#         width, height = image.size
#         if width != size[0] or height != size[1]:
#             image = image.resize(size)
#         return image

#     def __init__(self, sequence):
#         if sequence < 10:
#             sequence = f"0{sequence}"
#         else:
#             sequence = str(sequence)
#         fpath = f'datasets/data_odometry_csv/{sequence}.csv'
#         dataframe = pd.read_csv(fpath)
#         self.img_fpaths = dataframe["img_fpaths"].to_numpy()
#         self.odom_poses = dataframe[[f"odom_{i}" for i in range(1, 13)]].to_numpy()

#         self.received_invalid_image_sizes = set()

#     def __len__(self):
#         return len(self.img_fpaths)

#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
#         image = Image.open(self.img_fpaths[index])
#         image = self._maybe_resize_image(image, (1241, 376))
#         img_tensor = transform(image)

#         odom_pose = torch.from_numpy(self.odom_poses[index]).float()
    
#         return img_tensor, odom_pose

# def get_dataloaders(sequences, batch_size):
#     datasets = []
#     if isinstance(sequences, list):
#         for sequence in sequences:
#             datasets.append(KittiOdomDataset(sequence))
#     elif isinstance(sequences, tuple):
#         for sequence in range(*sequences):
#             datasets.append(KittiOdomDataset(sequence))
#     else:
#         raise Exception("Sequences is not list or tuple!")
    
#     return [DataLoader(dataset, batch_size=batch_size, pin_memory=True) for dataset in datasets]


# Code to verify dataloader is working
if __name__ == '__main__':
    dataset = get_batches(sequences=(0,8), batch_len=100)

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
