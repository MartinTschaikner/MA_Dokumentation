"""
This script plots the z projection by averaging data from the img volume data and the bmo data
"""
import numpy as np
import pandas as pd
from oct_img_reader import OctReadImg
from tkinter.filedialog import askopenfilename
import os.path
from ring_scan_from_img import geometric_median
import matplotlib.pyplot as plt

# define device for correct data reshaping
device = 'Cirrus/Topcon'

# hardcoded for Cirrus/Topcon img file
cube_size = np.array([6, 6, 2])

# show an "Open" dialog box and return the path to the selected file
file_name = askopenfilename(title='Select ' + device + ' img file')
file_path = os.path.dirname(file_name)
bmo_path = os.path.dirname(file_path) + '/bmo'

# get data to plot
oct_cirrus = OctReadImg(file_name, cube_size)

cell_info, cell_entries = oct_cirrus.get_cell_info()

vol_info, vol_size, cube = oct_cirrus.get_vol_info_and_size(cell_info, cell_entries)

num_a_scans, num_b_scans = oct_cirrus.get_volume_dimension(cube, vol_info, vol_size)

volume_data, volume_resolution = oct_cirrus.get_image_data(num_a_scans, num_b_scans)

z_project = np.average(volume_data, axis=0)

bmo_file_name = [f for f in os.listdir(bmo_path) if
                 f.find(cell_info[0]) != -1 and f.find(cell_info[4]) != -1 and f.find('.txt') != -1]

scan_id = cell_info[0] + '_' + cell_info[4]

file_name_bmo = bmo_path + '/' + bmo_file_name[0]

aspect = volume_resolution[1]/volume_resolution[2]

# get BMO data
bmo_crop = 15
bmo_data = pd.read_csv(file_name_bmo, sep=",", header=None)
bmo_points = np.zeros([bmo_data.shape[1], bmo_data.shape[0]], dtype=int)
bmo_points[:, 0] = bmo_data.iloc[0, :]
bmo_points[:, 1] = bmo_data.iloc[1, :] + bmo_crop
bmo_points[:, 2] = bmo_data.iloc[2, :]

bmo_center = np.concatenate((geometric_median(bmo_points[:, 0:2]), np.array([np.size(volume_data, 0)//2])))
bmo_center = bmo_center.astype('int')

# plot
font_size = 20
fig, ax = plt.subplots()
ax.imshow(z_project, interpolation='bicubic', aspect=aspect)
ax.plot(bmo_points[:, 0], bmo_points[:, 1], color='black', marker='.', linestyle='none',
        markersize=10, label='BMO points')
ax.plot(bmo_center[0], bmo_center[1], color='red', marker='.', linestyle='none', markersize=15, label='BMO center')
ax.set_title(device + ' SLO - Z projection - ' + cell_info[0] + '_' + cell_info[4], fontsize=font_size)
ax.legend(bbox_to_anchor=(1.35, 0.5), fontsize=font_size)
ax.set_xlabel('no. B scans [ ]', fontsize=font_size)
ax.set_ylabel('no. A scans [ ]', fontsize=font_size)
plt.show()
