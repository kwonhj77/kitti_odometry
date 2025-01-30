import cv2
import matplotlib.pyplot as plt 
import numpy as np
import time
import os

FOLDER_DIR = "./datasets/data_odometry_color/dataset/sequences/00/image_2/"
OUT_PATH = "./datasets/orb/results/"

# For each image frame, we only keep NUM_KEYPOINTS keypoints based on highest response values.
NUM_KEYPOINTS = 50

def main():

    # Initiate ORB
    orb = cv2.ORB_create()
    n_frames = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(OUT_PATH+"video.avi", fourcc, 30, (384, 300))
    for img_path in os.listdir(FOLDER_DIR):
        if img_path.endswith(".png"):
            n_frames += 1
            query_image = cv2.imread(FOLDER_DIR+img_path)
            gray_query_iamge = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

            keypoints, descriptors = orb.detectAndCompute(gray_query_iamge, None)

            # Try plotting all points without matching
            for keypoint in keypoints:
                pt = keypoint.pt
                cv2.circle(query_image,(int(pt[0]),int(pt[1])),3,(255,0,0),2)

            # Try plotting only top 50 responses
            sorted_keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
            for idx, keypoint in enumerate(sorted_keypoints):
                if idx > 50:
                    break
                pt = keypoint.pt
                cv2.circle(query_image,(int(pt[0]),int(pt[1])),2,(0,0,255),2)

            # Image needs to reshaped to even w/h values or else video will not write.
            query_image_resized = cv2.resize(query_image, (384, 300))
            videoWriter.write(query_image_resized)
            if n_frames % 200 == 0:
                print(f"frame {n_frames} processed")
    videoWriter.release()
if __name__ == '__main__':
    main()
    print("Exiting...")



