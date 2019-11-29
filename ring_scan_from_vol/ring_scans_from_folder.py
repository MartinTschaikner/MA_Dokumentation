"""
Example for a script to create a stack of ring scans from a given set of vol files.
"""

import numpy as np
from os import listdir
from os.path import isfile, join
import ring_scan_from_volume
import OCT_read
from tkinter.filedialog import askdirectory

# parameters for ring scan interpolation & plotting_boolean
radius = 1.75
filter_parameter = int(1)
number_circle_points = 768

# vol_files and file id for assignment between ring and volume scans
vol_path = askdirectory(title='Choose directory of Spectralis vol files')
vol_path = vol_path + '/'
vol_files = [f for f in listdir(vol_path) if isfile(join(vol_path, f))]

# bmo_files
bmo_path = askdirectory(title='Choose directory of Spectralis bmo files')
bmo_path = bmo_path + '/'
bmo_files = [f for f in listdir(bmo_path) if isfile(join(bmo_path, f))]

num_files = len(vol_files)
ring_scans = []
ring_scans_id = []

# computation for all ring scan files
for i in range(num_files):
    file_vol = vol_files[i]
    file_bmo = bmo_files[i]

    if file_vol.split('.')[0] != file_bmo.split('.')[0]:
        print('Wrong assignment of bmo and vol files!')

    # Get all the information about the input vol file
    oct_info_vol = OCT_read.OctRead(vol_path + file_vol)

    # Get the header of the input vol file
    header_vol = oct_info_vol.get_oct_hdr()

    scan_pos = str(header_vol['ScanPosition'])
    scan_pos_input = scan_pos[2:4]

    if scan_pos_input != str(file_bmo.split('.')[1]).split('_')[1]:
        print(i, scan_pos_input, str(file_bmo.split('.')[1]).split('_')[1])

    a = file_vol.split('.')[0]
    ring_scan_id = str(file_vol.split('.')[0]) + '_' + scan_pos_input

    # Get the b scan stack of the input vol file
    b_scan_stack = oct_info_vol.get_b_scans(header_vol)
    # Get the segmentation data of the input vol file
    seg_data_full = oct_info_vol.get_segmentation(header_vol)

    # compute interpolated grey values, ilm and rpe segmentation
    ring_scan_interpolated = \
        ring_scan_from_volume.RingScanFromVolume(header_vol, b_scan_stack, seg_data_full, bmo_path + file_bmo,
                                                 radius, number_circle_points, filter_parameter)

    # compute correct circle points to corresponding scan pattern (OS vs OD)
    circle_points = ring_scan_interpolated.circle_points_coordinates()

    shifts_B = [2, -1.5, 4.5, -1, 2.5]
    shifts_A = [-5, -5, -10, -3, -4]
    id_shift = [2, 3, 8, 9, 18]
    if i in id_shift:
        k = id_shift.index(i)
        circle_points[0] = circle_points[0] + shifts_B[k] * header_vol['Distance']
        circle_points[1] = circle_points[1] + shifts_A[k] * header_vol['ScaleX']
        print(i, ' - id')

    # compute interpolated grey values, ilm and rpe segmentation
    ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, remove_boolean = \
        ring_scan_interpolated.ring_scan_interpolation(circle_points)

    ring_scans.append(ring_scan_int)
    ring_scans_id.append(ring_scan_id)

ring_scans = np.dstack(ring_scans)
np.save('Spectralis_val_ring_scans.npy', ring_scans)
np.save('Spectralis_val_ring_scans_id.npy', ring_scans_id)
np.save('Spectralis_res.npy', [header_vol['Distance'], header_vol['ScaleX'], header_vol['ScaleZ']])
