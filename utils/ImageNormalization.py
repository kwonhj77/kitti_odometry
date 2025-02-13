import numpy as np
import cv2
import pykitti

BASE_DIR = r'C:\Users\Will Haley\Documents\GitHub\kitti_odometry\dataset'
SEQUENCES = [3, 4, 5, 6, 7, 10]
def main():
    std = []
    mean = []
    for seq in SEQUENCES:
        seq = f"0{seq}" if seq < 10 else str(seq)
        dataset = pykitti.odometry(BASE_DIR, seq)
        for idx in range(len(dataset)):
            img = dataset.get_cam2(idx) # PIL Image
            img = np.array(img)[:, :, ::-1].astype(np.float32) / 255.0

            mean.append(img.mean(axis=(0,1)))
            std.append(img.std(axis=(0,1)))

        #     cv2.imshow('img',img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     if idx > 10:
        #         break
        # break

    std = np.array(std).mean(axis=0)
    mean = np.array(mean).mean(axis=0)

    print(f"Mean: ")
    print(f"R: {mean[0]}")
    print(f"G: {mean[1]}")
    print(f"B: {mean[2]}")

    print(f"Std: ")
    print(f"R: {std[0]}")
    print(f"G: {std[1]}")
    print(f"B: {std[2]}")


if __name__ == '__main__':
    main()
    print("Exiting...")