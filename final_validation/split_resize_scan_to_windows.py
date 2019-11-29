import numpy as np
from skimage.transform import resize


def resize_stack(stack, new_height, new_width):

    resized_stack = np.empty((np.size(stack, 0), new_height, new_width, 1)).astype(dtype=np.uint8)
    print(np.size(stack, 0), 'ring scan(s) were resized to', new_height, 'x', new_width)

    for i in range(np.size(stack, 0)):
        resized_img = resize(stack[i, :, :, 0], (new_height, new_width), anti_aliasing=False, mode='reflect')
        resized_stack[i, :, :, 0] = np.around(255 * resized_img).astype(dtype=np.uint8)

    return resized_stack


def windowing(stack, data_name, window_width, window_height, stride_x, stride_y):
    """
    This function creates a new 4D stack with width/height of given window dimensions and
    corresponding strides over the original 4D stack

    :param stack:           4D data stack [number b scans, height, width of image, channels=1]
    :param data_name:       name data for string output in console
    :param window_width:    width of window in pixels
    :param window_height:   height of window in pixels (default: set to stack height to create stripes)
    :param stride_x:        stride in direction of width
    :param stride_y:        stride in direction of height (default: 1 - no stride necessary for  )
    :return:                new 4D stack of windows concatenated in direction of axis 0
    """

    # resize stack height to new height (512 for CNN) - no change in x direction
    stack = resize_stack(stack, window_height, np.size(stack, 2))

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



