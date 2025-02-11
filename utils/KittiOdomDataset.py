import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.RawDataParser import RawDataParser

BASE_DIR = r'C:\Users\Will Haley\Documents\GitHub\kitti_odometry\dataset'


class KittiOdomBatch(torch.utils.data.Dataset):
    def __init__(self, sequence, frame_start_idx, frame_end_idx, img_size, img_mean, img_std, stack_images):
        seq_fname = f"0{sequence}" if sequence < 10 else str(sequence)
        self.img_mean, self.img_std = img_mean, img_std

        assert len(img_size) == 3, f"image dims are invalid {img_size}"
        self.stack_images = stack_images

        self.img_size = (img_size[0], img_size[2], img_size[1])  # CxHxW to CxWxH
        # print(f"----------- \nSeq: {sequence} \n Batch frame range: {frame_start_idx}:{frame_end_idx-1}")

        batch = RawDataParser(BASE_DIR, seq_fname, frames=range(frame_start_idx, frame_end_idx, 1))

        # Adjust the rotations and positions so that the first frame in the batch is the global origin.
        init_rot, init_pos = self._split_rotation_and_position(batch.poses[0])
        assert np.linalg.det(init_rot) != 0, "Rotation matrix is singular! This shouldn't ever be the case AFAIK..."
        init_rot_inverse = np.linalg.inv(init_rot)

        self.rotation = []
        self.position = []
        for pose in batch.poses:
            rot, pos = self._split_rotation_and_position(pose)
            self.rotation.append(np.matmul(init_rot_inverse, rot))
            self.position.append(pos - init_pos)
        self.raw_dataset = batch

    def __len__(self):
        return len(list(self.raw_dataset.frames))
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.stack_images:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.img_mean+self.img_mean, std=self.img_std+self.img_std),
            ])
            # Get image from last time and combine as a HxWx6 Tensor
            if index == 0:
                img1 = self.raw_dataset.get_cam2(index)
            else:
                img1 = self.raw_dataset.get_cam2(index-1)
            img1 = self._maybe_resize_image(img1)
            img1 = np.array(img1)[:, :, ::-1].astype(np.float32) / 255.0

            img2 = self.raw_dataset.get_cam2(index)
            img2 = self._maybe_resize_image(img2)
            img2 = np.array(img2)[:, :, ::-1].astype(np.float32) / 255.0

            img_tensor = transform(np.concatenate((img1, img2), axis=2))
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.img_mean, std=self.img_std),
            ])
            img = self.raw_dataset.get_cam2(index)
            img = self._maybe_resize_image(img)
            img = np.array(img)[:, :, ::-1].astype(np.float32) / 255.0
            img_tensor = transform(img)
        rot = torch.from_numpy(self.rotation[index].flatten()).float()
        pos = torch.from_numpy(self.position[index].flatten()).float()
    
        return img_tensor, rot, pos
    
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

def _get_batches_from_sequence(sequence, sequence_len, params):
    batch_len = params['batch_len']
    img_size = params['img_size']
    img_mean = params['img_mean']
    img_std = params['img_std']
    stack_images = params['stack_input_images']
    batch_datasets = []
    for frame_start_idx in range(0, sequence_len, batch_len):
        frame_end_idx = frame_start_idx+batch_len if frame_start_idx+batch_len <= sequence_len else sequence_len
        batch_datasets.append(KittiOdomBatch(sequence, frame_start_idx, frame_end_idx, img_size, img_mean, img_std, stack_images))
    return batch_datasets

def get_batches(sequences, params):
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
        batches.extend(_get_batches_from_sequence(sequence=sequence, sequence_len=sequences_len[sequence], params=params))

    return [DataLoader(batch, batch_size=len(batch), pin_memory=True) for batch in batches]


# Code to verify dataloader is working
if __name__ == '__main__':
    params = dict()
    params['batch_len'] = 100
    params['img_size'] = [3, 376, 1241]
    params['img_mean'] = [0.36713704466819763, 0.3694778382778168, 0.3467831611633301]
    params['img_std'] = [0.31982553005218506, 0.310651570558548, 0.3016820549964905]
    params['stack_input_images'] = False
    dataset = get_batches(sequences=(0,1), params=params)
    X, rot, pos = next(iter(dataset[0]))
    print(X.shape)
    print(rot.shape)
    print(pos.shape)

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
