import os
import re
import numpy as np
import pandas as pd

# Run this file to generate .csv files for each sequence, which is used for the KittiOdomDataset creation.

IMG_ROOT_DIR = "./datasets/data_odometry_color/dataset/sequences/"
ODOM_ROOT_DIR = "./datasets/data_odometry_poses/dataset/poses/"
    
# Create a csv file containing the relative directory of images and the corresponding odometry value.
def create_label_csv():
    def get_image_fpaths(IMG_ROOT_DIR, sequence, selected_camera):
        img_dir = IMG_ROOT_DIR+sequence+"/"+selected_camera
        img_fpaths = []
        for img_fname in os.listdir(img_dir):
            if img_fname.endswith(".png"):
                img_fpaths.append(img_dir+'/'+img_fname)
        return np.array(img_fpaths)
    
    def get_odom_poses(ODOM_ROOT_DIR, sequence):
        chars_to_keep = ' -.'
        odom_fpath = ODOM_ROOT_DIR + sequence + ".txt"
        with open(odom_fpath) as f:
            lines = f.readlines()
            for line in lines:
                line = re.sub(r'[^\w'+chars_to_keep+']', '', line)
                line_split = line.split()
                odom_poses.append([float(i) for i in line_split])
        return np.array(odom_poses)


    labeled_sequence_range = (0, 10)
    unlabeled_sequence_range = (11, 21)
    selected_camera = "image_2"


    for sequence in range(labeled_sequence_range[0], labeled_sequence_range[1]+1):
        if sequence < 10:
            sequence = f"0{sequence}"
        else:
            sequence = str(sequence)

        # Find image filepaths
        img_fpaths = get_image_fpaths(IMG_ROOT_DIR, sequence, selected_camera)
        
        # Get odom 
        odom_poses = get_odom_poses(ODOM_ROOT_DIR, sequence)
        
        assert len(img_fpaths) == len(odom_poses), f"odom and image lengths mismatch!\n img: {len(img_fpaths)}\n odom: {len(odom_poses)}"

        dataframe = pd.DataFrame(data=odom_poses,
                                 columns=[f"odom_{i}" for i in range(1,13)])
        dataframe['img_fpaths'] = img_fpaths

        # Write sequence csv file
        fname = f'./datasets/data_odometry_csv/{sequence}.csv'
        dataframe.to_csv(fname)

    # for sequence in range(unlabeled_sequence_range[0], unlabeled_sequence_range[1]+1):
    #     # Not needed but added just for safety
    #     if sequence < 10:
    #         sequence = f"0{sequence}"
    #     else:
    #         sequence = str(sequence)

    #     # Find image filepaths
    #     img_fpaths = get_image_fpaths(IMG_ROOT_DIR, sequence, selected_camera)

    #     dataframe = pd.DataFrame(data=img_fpaths,
    #                              columns=["img_fpaths"])
        
    #     # Write sequence csv file
    #     fname = f'./datasets/data_odometry_csv/{sequence}.csv'
    #     dataframe.to_csv(fname)

if __name__ == '__main__':
    create_label_csv()
    print("Exiting...")