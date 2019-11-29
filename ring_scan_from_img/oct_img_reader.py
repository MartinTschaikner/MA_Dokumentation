"""

A class to read the raw OCT data (.vol file). This class has four different methods to read header file,
SLO image, b-scans, and corresponding segmentation.
"""

# import numerical python and structure packages
from struct import unpack
import numpy as np
import os


class OctReadImg:
    """

    This class will read the OCT volume file scanned by Cirrus.
    """

    def __init__(self, file_name, cube_size):
        """
        class/instance variables

        :param file_name: name of the volume file
        :type file_name: str
        :param cube_size: region of interest
        :type cube_size: ndarray
        """
        self.file_name = file_name
        self.cube_size = cube_size

    def get_cell_info(self):
        """

        :return:
        """
        name_file = self.file_name.split('.')[0]
        name_file = name_file.split('/')[-1]

        if name_file.find('clinic') == -1:
            cell_info = name_file.split('_')
        else:
            cell_info = name_file.split('_')

        cell_info_entry = cell_info[1].split(" ")

        if cell_info_entry[0] == 'Optic':
            cell_info_entry[0] = cell_info_entry[0] + " " + cell_info_entry[1]
            cell_info_entry.pop(1)

        return cell_info, cell_info_entry

    @staticmethod
    def get_vol_info_and_size(cell_info, cell_entries):
        """

        :param cell_entries:
        :param cell_info:
        :return:
        """

        vol_info = {'patient_id': cell_info[0],
                    'scan_type': cell_entries[0],
                    'scan_date': cell_info[2] + " " + cell_info[3],
                    'eye_side': cell_info[4]
                    }

        vol_size = cell_entries[2]
        cube = cell_info[7]

        return vol_info, vol_size, cube

    def get_volume_dimension(self, cube, volume_info, volume_size):
        """

        :param cube:
        :param volume_info:
        :param volume_size:
        :return:
        """
        if volume_info['scan_type'] != 'Macular' and volume_info['scan_type'] != 'Optic Disc':
            print("Error, cannot read Cirrus file: " + self.file_name + " must be a macular cube or optic disk scan.")

        volume_size = volume_size.split('x')
        num_ascans = int(volume_size[0])
        num_bscans = int(volume_size[1])

        if cube == 'hidef':
            num_bscans = 2
            if volume_info['scan_type'] == 'Macular':
                num_ascans = 1024
            elif volume_info['scan_type'] == 'Optic Disc':
                num_ascans = 1000

        return num_ascans, num_bscans

    def get_image_data(self, num_ascans, num_bscans):
        """

        :param num_ascans:
        :param num_bscans:
        :return:
        """
        # Read binary file
        f = open(self.file_name, "rb")
        scans_raw = np.fromfile(f, dtype=np.uint8)

        num_axial = len(scans_raw)/(num_ascans*num_bscans)
        if np.remainder(num_axial, 1) != 0.0:
            print("ERROR: problem reading file " + self.file_name)

        volume_data = np.reshape(scans_raw, (num_ascans, int(num_axial), num_bscans), order='F')
        volume_data = np.transpose(volume_data, (1, 0, 2))
        volume_data = np.flip(volume_data, 0)
        volume_data = np.flip(volume_data, 1)
        volume_data = np.flip(volume_data, 2)
        volume_resolution = np.array([self.cube_size[2]/(num_axial-1), self.cube_size[0]/(num_ascans-1),
                                      self.cube_size[1]/(num_bscans-1)])
        return volume_data, volume_resolution











