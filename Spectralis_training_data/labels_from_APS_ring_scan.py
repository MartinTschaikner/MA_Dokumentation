"""
This script creates an image stack plus corresponding labels.
Input data for Keras has to be a 4D array [number of images, height, width, channels = 1]
"""
import numpy as np
from os import listdir
from os.path import isfile, join
import OCT_read
from plots_labels_ring_scan import label_ring_scan_plots
import cv2

# for visual testing of computation
plotting_boolean = False
save_tiff_boolean = False

# ring_scan_files file id for assignment between ring and volume scans
ring_scan_path = 'ring_scan_validation/'
ring_scan_files = [f for f in listdir(ring_scan_path) if isfile(join(ring_scan_path, f))]

num_files = len(ring_scan_files)

# Get all the information about the first ring scan file
oct_info_ring_scan = OCT_read.OctRead(ring_scan_path + ring_scan_files[0])

# Get the header of the first ring scan file to calculate stack size
header_ring_scan = oct_info_ring_scan.get_oct_hdr()
height = header_ring_scan['SizeZ']
width = header_ring_scan['SizeX']

# placeholder image stack data
stack_ring_scan = np.empty((num_files, height, width, 1))
stack_ring_scan_label = np.empty((num_files, height, width, 1))

# fill placeholder with data
for i in range(num_files):

    file_ring_scan = ring_scan_files[i]

    # Get the all information about the input ring scan file
    oct_info_ring_scan = OCT_read.OctRead(ring_scan_path + file_ring_scan)

    # Get the header of the input ring scan file
    header_ring_scan = oct_info_ring_scan.get_oct_hdr()

    # Get the b scan stack of the input ring scan file
    b_scan_stack = oct_info_ring_scan.get_b_scans(header_ring_scan)

    # Get the segmentation data of the input ring scan file
    seg_data_full = oct_info_ring_scan.get_segmentation(header_ring_scan)

    # Get needed data
    ring_scan = b_scan_stack.reshape(header_ring_scan['SizeZ'], header_ring_scan['SizeX'])
    ilm_ring_scan = seg_data_full['SegLayers'][:, 0, :]
    rpe_ring_scan = seg_data_full['SegLayers'][:, 1, :]
    RNFL_ring_scan = seg_data_full['SegLayers'][:, 2, :]

    # checks for Nan in segmentation data
    check_ilm = np.isnan(ilm_ring_scan).astype('int')
    check_RNFL = np.isnan(RNFL_ring_scan).astype('int')
    check_rpe = np.isnan(rpe_ring_scan).astype('int')

    if np.sum(check_rpe) != 0:
        print(np.sum(check_rpe), 'Nan(s) in', i+1, '. RPE segmentation')

        # handle nans
        nan_ind = np.argwhere(check_rpe == 1)[:, 0]
        print(nan_ind)
        for j in range(len(nan_ind)):
            # replace nan with average of neighbours if only 1 nan
            if len(nan_ind) == 1:
                rpe_ring_scan[nan_ind[j]] = 0.5 * (rpe_ring_scan[nan_ind[j]+1] + rpe_ring_scan[nan_ind[j]-1])
            else:
                # replace nan with value of left neighbour
                rpe_ring_scan[nan_ind[j]] = rpe_ring_scan[nan_ind[j]-1]

    if np.sum(check_ilm) != 0:
        print('Nan ', i, 'ilm segmentation')

    if np.sum(check_RNFL) != 0:
        print('Nan', i, 'RNFL segmentation')
        nan_ind = np.argwhere(check_rpe == 1)

    # checks if ILM, RNFL, RPE are not intersecting
    check_1 = np.sum(ilm_ring_scan > RNFL_ring_scan).astype('int')
    check_2 = np.sum(RNFL_ring_scan > rpe_ring_scan).astype('int')
    if (check_1 + check_2) != 0:
        print(file_ring_scan, ': Intersection of segmentation detected!')

    # round segmentation data to nearest pixel
    ilm_ring_scan = np.around(ilm_ring_scan).astype('int')
    RNFL_ring_scan = np.around(RNFL_ring_scan).astype('int')
    rpe_ring_scan = np.around(rpe_ring_scan).astype('int')

    # create placeholder for label data
    ring_scan_label = np.zeros((header_ring_scan['SizeZ'], header_ring_scan['SizeX']))

    # fill label with segmentation classes
    for j in range(header_ring_scan['SizeX']):
        ring_scan_label[ilm_ring_scan[j, 0]:, j] = int(1)
        ring_scan_label[RNFL_ring_scan[j, 0]:, j] = int(2)
        ring_scan_label[rpe_ring_scan[j, 0]:, j] = int(3)

    # store data for labeled trainings examples
    stack_ring_scan[i, :, :, 0] = ring_scan
    stack_ring_scan_label[i, :, :, 0] = ring_scan_label

    if plotting_boolean is True:
        # plot ring scan of interpolated grey values and corresponding ilm and rpe segmentation
        label_ring_scan_plots(ring_scan, ring_scan_label, ilm_ring_scan, RNFL_ring_scan, rpe_ring_scan)

# save ring scans and masks as unsigned int8
np.save('ring_scan_stack', stack_ring_scan.astype(dtype=np.uint8))
np.save('ring_scan_label_stack', stack_ring_scan_label.astype(dtype=np.uint8))

# checking output visually
if save_tiff_boolean:
    # choose img from stack to visualize
    t = 2
    file_tiff_pre = ring_scan_files[t].split('.')[0]
    file_tiff_suf = '_test_img.tiff'
    file_tiff_suf_label = '_test_img_label.tiff'
    cv2.imwrite(file_tiff_pre + file_tiff_suf, stack_ring_scan[t, :, :, 0])
    cv2.imwrite(file_tiff_pre + file_tiff_suf_label, stack_ring_scan_label[t, :, :, 0])
    print('Test image saved as ' + file_tiff_pre + file_tiff_suf)
    print('Test image label saved as ' + file_tiff_pre + file_tiff_suf_label)
