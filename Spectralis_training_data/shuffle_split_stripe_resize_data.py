"""
This script shuffles the labeled ring scan stack, splits it into training, validation and test set. Then the images
within the sets are rescaled and split into overlapping stripes of the rescaled images to increase the number of
examples.

"""
import numpy as np
from skimage.transform import resize

# save stripe sets
save_boolean = True

# import data and augmentation
real_ring_scan = np.load('ring_scan_stack/ring_scan_stack.npy')
real_ring_scan_label = np.load('ring_scan_stack/ring_scan_label_stack.npy')
aug_ring_scan = np.load('aug_ring_scan_stack/aug_ring_scan_stack.npy')
aug_ring_scan_label = np.load('aug_ring_scan_stack/aug_ring_scan_label_stack.npy')

# concatenate data
ring_scan = np.concatenate((real_ring_scan, aug_ring_scan), axis=0)
ring_scan_label = np.concatenate((real_ring_scan_label, aug_ring_scan_label), axis=0)

# calculate split indices 60%, 20% and rest
num_img = np.size(ring_scan, 0)
num_train = np.around(0.6*num_img).astype('int')
ind_train = num_train
num_val = np.around(0.2*num_img).astype('int')
ind_val = num_train + num_val
num_test = num_img - num_train - num_val
ind_test = num_img

# shuffled input data
ind_rnd = np.random.permutation(num_img)
ring_scan = ring_scan[ind_rnd, :, :, :]
ring_scan_label = ring_scan_label[ind_rnd, :, :, :]

# split into train, validation and test set
rs_train = ring_scan[:ind_train, :, :, :]
rs_label_train = ring_scan_label[:ind_train, :, :, :]

rs_val = ring_scan[ind_train:ind_val, :, :, :]
rs_label_val = ring_scan_label[ind_train:ind_val, :, :, :]

rs_test = ring_scan[ind_val:, :, :, :]
rs_label_test = ring_scan_label[ind_val:, :, :, :]

# check split
if num_train == np.size(rs_train, 0) and num_val == np.size(rs_val, 0) and num_test == np.size(rs_test, 0):
    print('Check split of', num_img, 'ring scans ( train:', num_train, 'val:', num_val, 'test:', num_test, ') --- OK!')


def windowing(stack, data_name, window_width, window_height, stride_x, stride_y):
    """
    This function creates a new 4D stack with width/height of given window and
    corresponding strides of a given 4D stack

    :param stack:           4D data stack [number b scans, height, width of image, channels=1]
    :param data_name:       name data
    :param window_width:    width of window in pixels
    :param window_height:   height of window in pixels
    :param stride_x:        stride in direction of width
    :param stride_y:        stride in direction of height
    :return:                new 4D stack of windows concatenated in direction of axis 0
    """
    # original data sizes
    num_b_scans = np.size(stack, 0)
    height = np.size(stack, 1)
    width = np.size(stack, 2)

    # number of windows in both directions and decision parameter for missing windows
    num_x = (width - window_width) // stride_x + 1
    remainder_x = (width - window_width) % stride_x
    num_y = (height - window_height) // stride_y + 1
    remainder_y = (height - window_height) % stride_x

    # number of windows for placeholder of new stack. Missing windows added later
    num_windows = num_x * num_y * num_b_scans
    stack_of_windows = np.empty((num_windows, window_height, window_width, 1)).astype(dtype=np.uint8)

    # 2 for-loops to fill new data stack
    for j in range(num_y):
        for i in range(num_x):
            window_stack = stack[:, j*stride_y:j*stride_y + window_height, i*stride_x:i*stride_x + window_width, :]
            stack_of_windows[(j*num_x + i)*num_b_scans:(j*num_x + i + 1)*num_b_scans, :, :, :] = \
                window_stack

    # missing windows at right boundary of image added and update of num_windows for check
    if remainder_x != 0:
        num_windows = num_windows + num_y * num_b_scans
        for j in range(num_y):
            window_stack = stack[:, j*stride_y:j*stride_y + window_height, -window_width:, :]
            stack_of_windows = np.concatenate((stack_of_windows, window_stack), axis=0)

    # missing windows at lower boundary of image added and update of num_windows for check
    if remainder_y != 0:
        num_windows = num_windows + num_x * num_b_scans
        for i in range(num_x):
            window_stack = stack[:, -window_height:, i * stride_x:i * stride_x + window_width, :]
            stack_of_windows = np.concatenate((stack_of_windows, window_stack), axis=0)

    # missing last windows in lower right corner and update of num_windows for check
    if remainder_x != 0 and remainder_y != 0:
        num_windows = num_windows + num_b_scans
        window_stack = stack[:, -window_height:, -window_width:, :]
        stack_of_windows = np.concatenate((stack_of_windows, window_stack), axis=0)

    if num_windows == np.size(stack_of_windows, 0):
        print('Check number of windowed ' + data_name + ' examples:', np.size(stack_of_windows, 0), ' --- OK!')

    return stack_of_windows


def resize_stack(stack, new_height, new_width):

    resized_stack = np.empty((np.size(stack, 0), new_height, new_width, 1)).astype(dtype=np.uint8)
    print(np.size(stack, 0), 'ring scans are resized to', new_height, 'x', new_width)

    for i in range(np.size(stack, 0)):
        resized_img = resize(stack[i, :, :, 0], (new_height, new_width), anti_aliasing=True, mode='reflect')
        resized_stack[i, :, :, 0] = np.around(255 * resized_img).astype(dtype=np.uint8)

    return resized_stack


# window width, strides and height resize parameters
window_w = int(128)
if window_w > np.size(ring_scan, 2):
    window_w = np.size(ring_scan, 2)

x_stride = int(10)
y_stride = int(1)

height_rescale = 512

# rescaling and windowing the three different sets
training_data = resize_stack(rs_train, height_rescale, np.size(ring_scan, 2))
training_label = resize_stack(rs_label_train, height_rescale, np.size(ring_scan, 2))
training_data = windowing(training_data, 'training', window_w, height_rescale, x_stride, y_stride)
training_label = windowing(training_label, 'labeled training', window_w, height_rescale, x_stride, y_stride)

validation_data = resize_stack(rs_val, height_rescale, np.size(ring_scan, 2))
validation_label = resize_stack(rs_label_val, height_rescale, np.size(ring_scan, 2))
validation_data = windowing(validation_data, 'training', window_w, height_rescale, x_stride, y_stride)
validation_label = windowing(validation_label, 'labeled training', window_w, height_rescale, x_stride, y_stride)

test_data = resize_stack(rs_test, height_rescale, np.size(ring_scan, 2))
test_label = resize_stack(rs_label_test, height_rescale, np.size(ring_scan, 2))
test_data = windowing(test_data, 'training', window_w, height_rescale, x_stride, y_stride)
test_label = windowing(test_label, 'labeled training', window_w, height_rescale, x_stride, y_stride)

# save final data for training, validation and testing
if save_boolean:
    np.save('data/train/training_data_width_' + str(window_w) + '_stride_' + str(x_stride),
            training_data.astype(dtype=np.uint8))
    np.save('data/train/training_label_width_' + str(window_w) + '_stride_' + str(x_stride),
            training_label.astype(dtype=np.uint8))
    np.save('data/val/validation_data_width_' + str(window_w) + '_stride_' + str(x_stride),
            validation_data.astype(dtype=np.uint8))
    np.save('data/val/validation_label_width_' + str(window_w) + '_stride_' + str(x_stride),
            validation_label.astype(dtype=np.uint8))
    np.save('data/test/test_data_width_' + str(window_w) + '_stride_' + str(x_stride),
            test_data.astype(dtype=np.uint8))
    np.save('data/test/test_label_width_' + str(window_w) + '_stride_' + str(x_stride),
            test_label.astype(dtype=np.uint8))
