"""
This script creates an image stack plus corresponding labels.
Input data for Keras has to be a 4D array [number of images, height, width, channels = 1]
"""
import numpy as np
from os import listdir, path
from jason_marker_array_converter import json_to_array
from viewer_data_label import plot_data_label

device = 'Cirrus'
filter_parameter = 2

# data paths
ring_scan_stack_path = device + r'_ring_scan_stack/'
ring_scan_stack_file = ring_scan_stack_path + device + '_ring_scan_stack_f' + str(filter_parameter) + '.npy'
json_markers_path = r'json_marker_files/' + device + '_corrected_f2_final_data/'

# import data
ring_scan_stack = np.load(ring_scan_stack_file)
if len(ring_scan_stack.shape) == 3:
    ring_scan_stack = np.expand_dims(ring_scan_stack, axis=3)
json_markers = [f for f in listdir(json_markers_path)
                if f.find('.joctmarker') != -1]
# check import
if len(json_markers) == np.size(ring_scan_stack, 0):
    print('Check OK!')
else:
    print('Number of ring scans differs from number of segmentation data!')

ring_scan_label_stack = np.zeros(ring_scan_stack.shape)
segmentation = []
mask = []

# fill placeholder with data
for i in range(np.size(ring_scan_stack, 0)):

    i_jason_marker = [f for f in json_markers if f.split('_')[-2] == str(i)]

    if len(i_jason_marker) != 0:
        ind_new = int(i_jason_marker[0].split('_')[-2])
        mask.append(ind_new)
        marker_file = json_markers_path + i_jason_marker[0]

        # get segmentation data from marker file
        segmentation = json_to_array(marker_file)
        ilm_ring_scan = segmentation[0, :]
        RNFL_ring_scan = segmentation[1, :]
        rpe_ring_scan = segmentation[2, :]

        # round segmentation data to nearest pixel
        ilm_ring_scan = np.around(ilm_ring_scan).astype('int')
        RNFL_ring_scan = np.around(RNFL_ring_scan).astype('int')
        rpe_ring_scan = np.around(rpe_ring_scan).astype('int')

        # fill label with segmentation classes
        for j in range(np.size(segmentation, 1)):
            ring_scan_label_stack[ind_new, ilm_ring_scan[j]:, j, :] = int(1)
            ring_scan_label_stack[ind_new, RNFL_ring_scan[j]:, j, :] = int(2)
            ring_scan_label_stack[ind_new, rpe_ring_scan[j]:, j, :] = int(3)

ring_scan_stack_final = ring_scan_stack[mask, :, :, :]
ring_scan_label_stack = ring_scan_label_stack[mask, :, :, :]

np.save(ring_scan_stack_path + device +
        '_ring_scan_stack_f' + str(filter_parameter), ring_scan_stack_final.astype(dtype=np.uint8))
np.save(ring_scan_stack_path + device +
        '_ring_scan_label_stack_f' + str(filter_parameter), ring_scan_label_stack.astype(dtype=np.uint8))

plot_data_label(ring_scan_stack_final[:, :, :, 0], ring_scan_label_stack[:, :, :, 0], device)
