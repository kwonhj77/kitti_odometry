import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.RawDataParser import RawDataParser

BASE_DIR = r'./dataset'


class KittiOdomBatch(torch.utils.data.Dataset):
    def __init__(self, sequence, frame_start_idx, frame_end_idx, img_size, img_mean, img_std, stack_images): # Currently stack_images is not used
        seq_fname = f"0{sequence}" if sequence < 10 else str(sequence)
        self.img_mean, self.img_std = img_mean, img_std

        assert len(img_size) == 3, f"image dims are invalid {img_size}"
        assert img_size[0] == 3, f"image channels are invalid {img_size[0]}"

        self.img_size = (img_size[0], img_size[2], img_size[1])  # CxHxW to CxWxH
        # print(f"----------- \nSeq: {sequence} \n Batch frame range: {frame_start_idx}:{frame_end_idx-1}")

        batch = RawDataParser(BASE_DIR, seq_fname, frames=range(frame_start_idx, frame_end_idx, 1))

        self.rotation = []
        self.position = []
        for pose in batch.poses:
            rot, pos = self._split_rotation_and_position(pose)
            self.rotation.append(rot)
            self.position.append(pos)
        self.raw_dataset = batch

    def __len__(self):
        return len(list(self.raw_dataset.frames))
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.img_mean, std=self.img_std),
        ])

        # Get current and t-1 images
        img_curr = self.raw_dataset.get_cam2(index)
        img_curr = self._maybe_resize_image(img_curr)
        img_curr = np.array(img_curr)[:, :, ::-1].astype(np.float32) / 255.0
        img_curr_tensor = transform(img_curr)

        if index == 0:
            img_prev = self.raw_dataset.get_cam2(index)
        else:
            img_prev = self.raw_dataset.get_cam2(index-1)
        img_prev = self._maybe_resize_image(img_prev)
        img_prev = np.array(img_prev)[:, :, ::-1].astype(np.float32) / 255.0
        img_prev_tensor = transform(img_prev)

        # Get current and t-1 rotation
        rot_curr = self.rotation[index]
        if index == 0:
            rot_prev_inv = self.rotation[index]
        else:
            rot_prev_inv = self.rotation[index-1]
        assert np.linalg.det(rot_prev_inv) != 0, "Rotation matrix is singular! This shouldn't ever be the case AFAIK..."
        rot_prev_inv = np.linalg.inv(rot_prev_inv)
        rot_curr = np.matmul(rot_curr, rot_prev_inv)

        # Get current and t-1 position
        pos_curr = self.position[index]
        if index == 0:
            pos_prev = self.position[index]
        else:
            pos_prev = self.position[index-1]
        pos_curr = pos_curr - pos_prev

        rot_euler = self._convert_euler_angle(rot_curr)
        rot = torch.from_numpy(rot_euler.flatten()).float()
        pos = torch.from_numpy(pos_curr.flatten()).float()

        # Get metadata
        seq_index = int(self.raw_dataset.sequence)
        timestamp = np.float64(self.raw_dataset.timestamps[index].total_seconds())
    
        return img_prev_tensor, img_curr_tensor, rot, pos, seq_index, timestamp
    
    def _maybe_resize_image(self, image):
        size = self.img_size[1:]
        # size is (width, height)
        width, height = image.size
        if width != size[0] or height != size[1]:
            image = image.resize(size)
        return image
    
    def _split_rotation_and_position(self, pose):
        assert isinstance(pose, np.ndarray) and pose.shape == (4,4)
        rot = pose[:3, :3]
        pos = pose[:3, -1]
        return rot, pos

    def _convert_euler_angle(self, rot):
        return Rotation.from_matrix(rot).as_euler('xyz', degrees=False)

def get_dataloader(sequences, params, shuffle):
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
    datasets = []
    if isinstance(sequences, tuple):
        sequences = range(*sequences)

    for sequence in sequences:
        sequence = KittiOdomBatch(sequence, 0, sequences_len[sequence], params['img_size'], params['img_mean'], params['img_std'], params['stack_input_images'])
        datasets.append(sequence)
    dataloader = DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=params['batch_len'], shuffle=shuffle, num_workers=0)

    return dataloader


# Code to verify dataloader is working
if __name__ == '__main__':
    params = dict()
    params['batch_len'] = 100
    params['img_size'] = [3, 376, 1241]
    params['img_mean'] = [0.36713704466819763, 0.3694778382778168, 0.3467831611633301]
    params['img_std'] = [0.31982553005218506, 0.310651570558548, 0.3016820549964905]
    params['stack_input_images'] = True
    dataloader = get_dataloader(sequences=(0,1), params=params, shuffle=True)
    X_prev, X_curr, rot, pos, seq, timestamp = next(iter(dataloader))
    print(X_prev.shape)
    print(X_curr.shape)
    print(rot.shape)
    print(pos.shape)
    print(seq)
    print(timestamp)

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
