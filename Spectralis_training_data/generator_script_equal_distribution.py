"""
Augmentation of trainings data.
"""

import keras
import numpy as np
from matplotlib import pyplot as plt

data_gen_args = dict(horizontal_flip=True,
                     zoom_range=[0.85, 1],
                     shear_range=8,
                     rotation_range=8,
                     height_shift_range=100,
                     fill_mode='reflect')


# initialize generator
img_gen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_gen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# load original labeled data
# X = np.load('Grid_img/Grid_img.npy')
# y = np.load('Grid_img/Grid_img_label.npy')
X = np.load('ring_scan_stack/ring_scan_stack.npy')
y = np.load('ring_scan_stack/ring_scan_label_stack.npy')

# boolean for saving, plotting and deleting if labels are cut off at top image margin
save_boolean = True
plotting_boolean = False
delete_aug = False

# number of used images and batch size
num_img = np.size(X, 0)
batch_size = 5
X = X[:num_img, :, :, :]
y = y[:num_img, :, :, :]

# parameters for plotting
noe = np.size(X, 2)
num_rows = np.size(X, 1)

# initialize data stacks
total_aug_img_stack = np.empty((1, np.size(X, 1), np.size(X, 2), 1)).astype(dtype=np.uint8)
total_aug_mask_stack = np.empty((1, np.size(X, 1), np.size(X, 2), 1)).astype(dtype=np.uint8)
total_num_aug = 0

# create augmented data and corresponding label with changing seed per loop
for j in range(num_img):

    for k in range(batch_size):
        seed = (k+1)*(j+1) + 1
        # original scan gets augmented batch_size times
        gen_img = img_gen.flow(X[j:j+1, :, :, :], batch_size=1, seed=seed)
        gen_mask = mask_gen.flow(y[j:j+1, :, :, :], batch_size=1, seed=seed)

        image_stack = gen_img[0].astype(dtype=np.uint8)
        mask_stack = gen_mask[0].astype(dtype=np.uint8)

        # plotting function
        if plotting_boolean:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
            ax1.imshow(X[j, :, :, 0], cmap='gray', vmin=0, vmax=255)
            ax1.set_title('Original ring scan', pad=22)
            ax1.title.set_size(25)
            ax1.set_xlabel('number of A scans [ ]', labelpad=18)
            ax1.set_ylabel('Z axis [ ]', labelpad=18)
            ax1.xaxis.label.set_size(20)
            ax1.yaxis.label.set_size(20)

            ax2.imshow(image_stack[0, :, :, 0], cmap='gray', vmin=0, vmax=255)
            ax2.set_title('Augmented ring scan', pad=22)
            ax2.title.set_size(25)
            ax2.set_xlabel('number of A scans [ ]', labelpad=18)
            ax2.set_ylabel('Z axis [ ]', labelpad=18)
            ax2.xaxis.label.set_size(20)
            ax2.yaxis.label.set_size(20)

            ax3.imshow(mask_stack[0, :, :, 0], cmap='gray', vmin=0, vmax=3, extent=(0, noe, num_rows, 0))
            ax3.set_title('Augmented ring scan Label', pad=22)
            ax3.title.set_size(25)
            ax3.set_xlabel('number of A scans [ ]', labelpad=18)
            ax3.xaxis.label.set_size(20)
            ax3.yaxis.set_major_locator(plt.NullLocator())
            plt.show()

        # concatenate data into one stack
        total_aug_img_stack = np.concatenate((total_aug_img_stack, image_stack), axis=0)
        total_aug_mask_stack = np.concatenate((total_aug_mask_stack, mask_stack), axis=0)

# delete first initialization image from total stack
total_aug_img_stack = total_aug_img_stack[1:, :, :, :]
total_aug_mask_stack = total_aug_mask_stack[1:, :, :, :]

# define some parameters and placeholder for deleting augmentations if wished
total_num_aug = num_img*batch_size
valid_num_aug = total_num_aug
del_ind = np.zeros(total_num_aug).astype('int')

# calculate intersecting augmentations
for t in range(num_img*batch_size):
    if np.sum(total_aug_mask_stack[t, 1, :, 0]) != 0:
        valid_num_aug -= 1
        del_ind[t] = 1

# indices could include mirrored augmentations that definitely have to be removed
del_ind = np.argwhere(del_ind == 1)[:, 0]
# placeholder for indices of augmentations with mirror labels
mirror_ind = np.zeros(del_ind.shape)
mirror_num = 0

# check for negative differences in label changes for first and last image column
for i in range(len(del_ind)):
    check_left = np.diff(np.int8(total_aug_mask_stack[del_ind[i], :, 0, 0]))
    check_right = np.diff(np.int8(total_aug_mask_stack[del_ind[i], :, -1, 0]))
    check = np.argwhere(check_left < 0).size + np.argwhere(check_right < 0).size
    if check != 0:
        mirror_ind[i] = 1
        mirror_num += 1

    plot = False
    if plot:
        fig, ax = plt.subplots(ncols=1)
        ax.imshow(total_aug_mask_stack[del_ind[i], :, :, 0], cmap='gray', vmin=0, vmax=3)
        ax.set_title('Augmented label - check: ' + str(check), pad=22)
        ax.title.set_size(25)
        ax.set_xlabel('number of A scans [ ]', labelpad=18)
        ax.set_ylabel('Z axis [ ]', labelpad=18)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        plt.show()

# indices of augmentations with mirror effects
mirror_ind = np.argwhere(mirror_ind == 1)[:, 0]

print(valid_num_aug, 'augmentations don´t intersect with upper image margin!\n', total_num_aug - valid_num_aug,
      'augmentations do intersect!\n', mirror_num, 'augmentations show mirror effects and are not used!')

# if true, delete augmentations that intersect with top image margin
if delete_aug:
    print(valid_num_aug, 'augmentations were saved!', '\n', total_num_aug - valid_num_aug,
          'augmentations were not valid and deleted!')
    print('Indices of deleted augmentations:', del_ind)
    total_num_aug = valid_num_aug
else:
    del_ind = del_ind[mirror_ind]
    total_num_aug = total_num_aug - mirror_num

print('A total of', total_num_aug, 'augmentations have been created!')

valid_aug_img_stack = np.delete(total_aug_img_stack, del_ind, axis=0)
valid_aug_mask_stack = np.delete(total_aug_mask_stack, del_ind, axis=0)

if np.size(valid_aug_img_stack, 0) == total_num_aug:
    print('Check OK!')

if save_boolean:
    # save all except the first empty image used for initialization
    np.save('aug_ring_scan_stack/aug_ring_scan_stack', valid_aug_img_stack)
    np.save('aug_ring_scan_stack/aug_ring_scan_label_stack', valid_aug_mask_stack)
