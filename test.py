import pykitti
import numpy as np
import matplotlib.pyplot as plt

# basedir = r'C:\Users\Will Haley\Documents\GitHub\kitti_odometry\dataset'
# sequence = '00'

# dataset = pykitti.odometry(basedir, sequence, frames=range(0, 50, 5))
# # Grab some data
# second_pose = dataset.poses[1]
# # first_gray = next(iter(dataset.gray))
# # first_cam1 = next(iter(dataset.cam1))
# first_rgb = dataset.get_rgb(0)
# first_cam2 = dataset.get_cam2(0)
# third_velo = dataset.get_velo(2)

# # Display some of the data
# np.set_printoptions(precision=4, suppress=True)
# print('\nSequence: ' + str(dataset.sequence))
# print('\nFrame range: ' + str(dataset.frames))

# # print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
# print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

# print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
# print('\nSecond ground truth pose:\n' + str(second_pose))

# f, ax = plt.subplots(2, 2, figsize=(15, 5))
# # ax[0, 0].imshow(first_gray[0], cmap='gray')
# # ax[0, 0].set_title('Left Gray Image (cam0)')

# # ax[0, 1].imshow(first_cam1, cmap='gray')
# # ax[0, 1].set_title('Right Gray Image (cam1)')

# ax[1, 0].imshow(first_cam2)
# ax[1, 0].set_title('Left RGB Image (cam2)')

# ax[1, 1].imshow(first_rgb[1])
# ax[1, 1].set_title('Right RGB Image (cam3)')
# plt.show()
# print("Exiting...")


basedir = r'C:\Users\Will Haley\Documents\GitHub\kitti_odometry\dataset'
sequence = '00'

dataset = pykitti.odometry(basedir, sequence, frames=range(0, 4540, 5))
# Remove last row from poses
poses = dataset.poses

# Copied code
ground_truth = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    ground_truth[i] = np.array(poses[i][:3]).reshape((3, 4))
# %matplotlib widget
fig = plt.figure(figsize=(7,6))
traj = fig.add_subplot(111, projection='3d')
traj.plot(ground_truth[:,:,3][:,0], ground_truth[:,:,3][:,1], ground_truth[:,:,3][:,2])
traj.set_xlabel('x')
traj.set_ylabel('y')
traj.set_zlabel('z')
plt.show()