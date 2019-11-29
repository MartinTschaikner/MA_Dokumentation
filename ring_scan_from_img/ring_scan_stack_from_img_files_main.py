"""
This script creates interpolated ring scans from given img vol files and bmo files.
"""

from oct_img_reader import OctReadImg
from ring_scan_from_img import RingScanFromImg
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tkinter.filedialog import askdirectory

# define device for correct data reshaping
device = 'Cirrus'

# define saving path for stack
save_path = device + r'_ring_scan_stack/'

# hardcoded for Cirrus/Topcon img file
cube_size = np.array([6, 6, 2])

# parameters
save_npy_boolean = True
radius = 1.75
number_circle_points = 768
filter_parameter = 2

# show an "Open" dialog box and return the path of the img files to convert
title = 'Choose folder with ' + device
file_path = askdirectory(title=title + ' img data!')
bmo_path = os.path.dirname(file_path) + '/bmo'

file_names = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.find('.img') != -1]

# initialize stack variable and id
ring_scan_stack = []
volume_resolution = []
ring_scans_id = [['ring scan #', 'ID', 'BMO center x', 'BMO center y']]

for i in range(len(file_names)):
    file_name = file_names[i]
    print(i)
    # import img data from file name
    oct_img = OctReadImg(file_path + '/' + file_name, cube_size)
    cell_info, cell_entries = oct_img.get_cell_info()
    vol_info, vol_size, cube = oct_img.get_vol_info_and_size(cell_info, cell_entries)
    num_a_scans, num_b_scans = oct_img.get_volume_dimension(cube, vol_info, vol_size)
    volume_data, volume_resolution = oct_img.get_image_data(num_a_scans, num_b_scans)

    # change resolution vector for each device
    if device == 'Cirrus':
        volume_resolution = np.roll(volume_resolution, 2)
        # get corresponding bmo file for ring scan interpolation center
        bmo_file_name = [f for f in os.listdir(bmo_path) if
                         f.find(cell_info[0]) != -1 and f.find(cell_info[4]) != -1 and f.find('.txt') != -1]
        # scan_id = cell_info[5]
        scan_id = cell_info[0] + '_' + cell_info[4]
    else:
        volume_resolution[0], volume_resolution[2] = volume_resolution[2], volume_resolution[0]
        bmo_file_name = [f for f in os.listdir(bmo_path) if
                         f.find(cell_info[0]) != -1 and f.find(cell_info[4]) != -1 and f.find('.txt') != -1]
        # scan_id = cell_info[0].split('-')[-1] + '_' + cell_info[2] + '_' + cell_info[3]
        scan_id = cell_info[0] + '_' + cell_info[4]

    if len(bmo_file_name) != 0:

        file_name_bmo = bmo_path + '/' + bmo_file_name[0]

        # compute interpolated ring scan
        ring_scan_from_img = RingScanFromImg(volume_data, file_name_bmo, volume_resolution, cell_info[4], radius,
                                             number_circle_points, filter_parameter)
        circle_points = ring_scan_from_img.circle_points_coordinates()

        if device == 'Topcon':
            shifts_B = [0, 4, 0, -1.5, -3]
            shifts_A = [-10, 1.5, -15, 1.5, 7]
            id_shift = [1, 5, 6, 7, 18]
        else:
            shifts_B = [3, 2, -3]
            shifts_A = [0, 0, -3]
            id_shift = [16, 17, 15]

        if i in id_shift:
            k = id_shift.index(i)
            circle_points[0] = circle_points[0] + shifts_B[k] * volume_resolution[0]
            circle_points[1] = circle_points[1] + shifts_A[k] * volume_resolution[1]
            print(scan_id)

        ring_scan_int = ring_scan_from_img.ring_scan_interpolation(circle_points)

        # add dimension to stack up
        ring_scan_int = np.expand_dims(ring_scan_int, axis=0)

        # initialize ring scan stack
        if i == 0:
            ring_scan_stack = ring_scan_int
        else:
            ring_scan_stack = np.concatenate((ring_scan_stack, ring_scan_int), axis=0)

        plot = False
        if plot:
            # plot to check
            noe = number_circle_points
            fig, ax1 = plt.subplots(ncols=1)

            ax1.imshow(ring_scan_int[0, :, :], cmap='gray', vmin=0, vmax=255)
            ax1.set_title('Interpolated ring scan from img file', pad=22)
            ax1.title.set_size(25)
            ax1.set_xlabel('number of A scans [ ]', labelpad=18)
            ax1.set_ylabel('Z axis [ ]', labelpad=18)
            ax1.xaxis.label.set_size(20)
            ax1.yaxis.label.set_size(20)

            plt.show()

        bmo_center = 0.5 * (circle_points[:, 0] + circle_points[:, number_circle_points//2])

        ring_scans_id.append([i, scan_id,
                              bmo_center[0]//volume_resolution[0], bmo_center[1]//volume_resolution[1]])

    else:
        print('No bmo data found - ring scan interpolation skipped')

# save interpolated ring scan stack and id text file
if save_npy_boolean:
    np.save(save_path + device + '_validation_ring_scan_stack_f' + str(filter_parameter) + '.npy',
            ring_scan_stack.astype(dtype=np.uint8))

    np.save(save_path + device + '_vol_res.npy', volume_resolution)

    print('Interpolated ring scan stack saved to ' + device + '_validation_ring_scan_stack.npy!')

    with open(save_path + device + '_validation_ring_scans_id_f' + str(filter_parameter) + '.txt', 'w') as f:
        for i in range(np.size(ring_scans_id, 0)):
            if i != 0:
                f.write('\n')
            for item in ring_scans_id[i]:
                f.write("%s" % item + '\t')
