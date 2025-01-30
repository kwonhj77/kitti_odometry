import torch
import os
import cv2

FOLDER_DIR = "./datasets/data_odometry_color/dataset/sequences/00/image_2/"

def inference():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load Inference Images
    imgs = []
    n_frame = 0
    for img_path in os.listdir(FOLDER_DIR):
        if img_path.endswith(".png"):
            n_frame += 1
            imgs.append(cv2.imread(FOLDER_DIR+img_path))
            if n_frame > 100:
                break
    print(f"{n_frame} images loaded.")
    # Inference
    results = model(imgs)

    results.print()
    results.save()

def convert_imgs_to_video():
    fpath = './runs/detect/exp/'
    out_path = "./"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(out_path+"yolov5.avi", fourcc, 30, (384, 300))

    n_frame = 0
    # TODO - the images in this are sorted weirdly (39, 4, 40, 41, ...). Fix this.
    for img_path in os.listdir(fpath):
        if img_path.endswith(".jpg"):
            print(img_path)
            n_frame += 1
            img = cv2.imread(fpath+img_path)
            img_resized = cv2.resize(img, (384, 300))
            videoWriter.write(img_resized)
    videoWriter.release()
    print("Video creation complete!")






if __name__ == '__main__':
    # inference()
    convert_imgs_to_video()
    print("Exiting...")