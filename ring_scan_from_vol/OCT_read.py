""" A class to read the raw OCT data (.vol file). This class has four different methods to read header file, 
SLO image, Bscans, and corresponding segmentation.
@input Param = name of the vol file
@author = Sunil K. Yadav (SKY)
@Date = 07.08.2018"""

# import numerical python and structure packages
from struct import unpack
import numpy as np
import os


class OctRead:
    def __init__(self, file_name):
        self.file_name = file_name

    """Read the header of the .vol file and return it as a Python dictionary."""
    def get_oct_hdr(self):

        # Read binary file
        with open(self.file_name, mode='rb') as file_name:
            file_content = file_name.read()
            version,\
            size_x,\
            num_b_scans,\
            size_z, \
            scale_x, \
            distance, \
            scale_z, \
            size_x_slo, \
            size_y_slo, \
            scale_x_slo,\
            scale_y_slo, \
            field_size_slo, \
            scan_focus, \
            scan_position, \
            exam_time, \
            scan_pattern, \
            b_scan_hdr_size, \
            id,\
            reference_id, \
            pid, \
            patient_id, \
            padding, \
            dob, \
            vid, \
            visit_id, \
            visit_date, \
            grid_type, \
            grid_offset,\
            grid_type_1, \
            grid_offset_1, \
            prog_id, \
            spare = unpack(
            "=12siiidddiiddid4sQii16s16si21s3sdi24sdiiii34s1790s", file_content[:2048])

        # Read raw hdr
        # Format hdr properly
        hdr = {'Version': version.rstrip(),
               'SizeX': size_x,
               'NumBScans': num_b_scans,
               'SizeZ': size_z,
               'ScaleX': scale_x,
               'Distance': distance,
               'ScaleZ': scale_z,
               'SizeXSlo': size_x_slo,
               'SizeYSlo': size_y_slo,
               'ScaleXSlo': scale_x_slo,
               'ScaleYSlo': size_y_slo,
               'FieldSizeSlo': field_size_slo,
               'ScanFocus': scan_focus,
               'ScanPosition': scan_position.rstrip(),
               'ExamTime': exam_time,
               'ScanPattern': scan_pattern,
               'BScanHdrSize': b_scan_hdr_size,
               'ID': id.rstrip(),
               'ReferenceID': reference_id.rstrip(),
               'PID': pid,
               'PatientID': patient_id.rstrip(),
               'DOB': dob,
               'VID': vid,
               'VisitID': visit_id.rstrip(),
               'VisitDate': visit_date,
               'GridType': grid_type,
               'GridOffset': grid_offset,
               'GridType1': grid_type_1,
               'GridOffset1': grid_offset_1,
               'ProgID': prog_id.rstrip()}
        return hdr

    """Read the EDTRS thickness grid information from the vol file."""
    def get_thickness_grid(self, hdr):
        # offset for the corresponding data
        grid_offset = hdr['GridOffset']

        # Type of the EDTRS grid with different diameters
        grid_type = hdr['GridType']

        # if zero then grid information are not available
        if grid_type != 0:

            # Read the vol file as a binary file from the offset point
            f = open(self.file_name, "rb")
            f.seek(grid_offset, os.SEEK_SET)

            # Read the thickness grid related information from vol file
            thickness_type = np.fromfile(f, dtype='int32', count=1)
            thickness_dia = np.fromfile(f, dtype='double', count=3)
            thickness_center_pos = np.fromfile(f, dtype='double', count=2)
            thickness_center_thick = np.fromfile(f, dtype='float32', count=1)
            thickness_min_center_thick = np.fromfile(f, dtype='float32', count=1)
            thickness_max_center_thick = np.fromfile(f, dtype='float32', count=1)
            thickness_total_vol = np.fromfile(f, dtype='float32', count=1)
            thickness_sector = np.zeros([9, 2], dtype='float')
            for i in range(9):
                thickness_sector[i, 0] = np.fromfile(f, dtype='float32', count=1)
                thickness_sector[i, 1] = np.fromfile(f, dtype='float32', count=1)

            # Format the data properly
            thickness_grid_info = {'Type': thickness_type,
                              'Diameter': thickness_dia,
                              'CenterPos': thickness_center_pos,
                              'CenterThick': thickness_center_thick,
                              'MinCentralThick': thickness_min_center_thick,
                              'MaxCentralThick': thickness_max_center_thick,
                              'TotalVolume': thickness_total_vol,
                              'ThicknessSectors': thickness_sector}

        return thickness_grid_info

    """Read the SLO image stored in the  .vol file and returns it as a Numpy array.
    @input param, hdr = header file, which is the output of the first method of the class. """
    def get_slo_image(self, hdr):
        # Size of the SLO image
        size_x_slo = hdr['SizeXSlo']
        size_y_slo = hdr['SizeYSlo']
        slo_size = size_x_slo * size_y_slo

        # Offset value before the SLO image
        slo_offset = 2048

        # Read the vol file as a binary file from the offset point
        f = open(self.file_name, "rb")
        f.seek(slo_offset, os.SEEK_SET)

        # Convert the data into proper format
        slo_img = np.fromfile(f, dtype=np.uint8, count=slo_size)

        # Convert the read data into matrix (array) form and return the image value
        slo_img = slo_img.reshape(size_x_slo, size_y_slo)
        return slo_img

    """method to read the bscan images from the raw vol file.
    @input param, hdr = header file, which is the output of the first method of the class"""
    def get_b_scans(self, hdr):
        # Read the size of the 3D image stack.
        size_x = hdr['SizeX']
        size_y = hdr['NumBScans']
        size_z = hdr['SizeZ']

        # Create and empty 3D array to store the bscans stack.
        b_scans_stack = np.zeros([size_z, size_x, size_y], dtype='float')

        # Get the size of the SLO image and also the size of the bscan header.
        size_x_slo = hdr['SizeXSlo']
        size_y_slo = hdr['SizeYSlo']
        b_scan_hdr_size = hdr['BScanHdrSize']

        # Size of a single bscan.
        loc_size = (size_x * size_z)

        # read the file as a binary file.
        f = open(self.file_name, "rb")

        # Get the data regarding each bscan.
        for i in range(size_y):
            # Offset value for each bscan.
            b_scan_offset = 2048 + b_scan_hdr_size + (size_x_slo*size_y_slo) + (i*(b_scan_hdr_size+size_x*size_z*4))

            # Read the corresponding bscan data.
            f.seek(b_scan_offset, os.SEEK_SET)
            b_scan = np.fromfile(f, '<f', count=loc_size)

            # Reshape in matrix format.
            b_scan = np.asarray(b_scan, 'f')
            b_scan = b_scan.reshape(size_z, size_x)

            # Convert into standard pixel data.
            b_scan = np.multiply((b_scan ** 0.25), 255)

            # Store in the stack.
            b_scans_stack[:, :, i] = b_scan
        return b_scans_stack

    """method to read the segmentation corresponding to the different layers of the retina from the raw vol file.
       @input param, hdr = header file, which is the output of the first method of the class"""
    def get_segmentation(self, hdr):

        # Size of the bscan stack
        size_x = hdr['SizeX']
        size_y = hdr['NumBScans']
        size_z = hdr['SizeZ']

        # Size of the SLO image
        size_x_slo = hdr['SizeXSlo']
        size_y_slo = hdr['SizeYSlo']
        b_scan_hdr_size = hdr['BScanHdrSize']

        # Binary data
        f = open(self.file_name, "rb")

        # Arrays to keep the boundary information of the segmentation
        b_scan_hdr_size_array = np.zeros([size_y, 1], dtype='int32')
        start_x = np.zeros([size_y, 1], dtype='double')
        start_y = np.zeros([size_y, 1], dtype='double')
        end_x = np.zeros([size_y, 1], dtype='double')
        end_y = np.zeros([size_y, 1], dtype='double')
        num_seg = np.zeros([size_y, 1], dtype='int32')
        off_seg = np.zeros([size_y, 1], dtype='int32')
        quality = np.zeros([size_y, 1], dtype='float32')
        shift = np.zeros([size_y, 1], dtype='int32')

        # create an empty stack for segmentation line
        b_scans_segments = np.zeros([size_x, 17, size_y], dtype='float')

        # read the segmentation lines
        for i in range(size_y):
            # offset regarding the each scan and then read
            seg_offset = 12 + 2048 + (size_x_slo * size_y_slo) + (i*(b_scan_hdr_size + size_x*size_z*4))
            f.seek(seg_offset, os.SEEK_SET)

            # Boundary points of the segments
            b_scan_hdr_size_array[i] = np.fromfile(f, dtype='int32', count=1)
            start_x[i] = np.fromfile(f, dtype='double', count=1)
            start_y[i] = np.fromfile(f, dtype='double', count=1)
            end_x[i] = np.fromfile(f, dtype='double', count=1)
            end_y[i] = np.fromfile(f, dtype='double', count=1)

            # number of segment layers
            num_seg[i] = np.fromfile(f, dtype=np.int32, count=1)

            # Offset to read the segment
            off_seg[i] = np.fromfile(f, dtype='int32', count=1)

            # Other params
            quality[i] = np.fromfile(f, dtype='float32', count=1)
            shift[i] = np.fromfile(f, dtype='int32', count=1)
            spare = np.fromfile(f, dtype='int8', count=192)
            # print('spare' + spare[i])

            seg_offset = 256 + 2048 + (size_x_slo * size_y_slo) + (i * (b_scan_hdr_size + size_x * size_z * 4))
            f.seek(seg_offset, os.SEEK_SET)
            first_seg = np.fromfile(f, '<f', count=num_seg[i, 0]*size_x)

            # maximum intensity of the segmentation and total number of points
            seg_max = (max(first_seg))
            num_points = num_seg[i, 0]*size_x

            # Remove outliers
            for k in range(num_points):
                if first_seg[k] == seg_max:
                    first_seg[k] = float('nan')

            # Store in the stack
            for j in range(num_seg[i, 0]):
                b_scans_segments[:, j, i] = first_seg[j*size_x:(j+1)*size_x]

        # Format the data properly
        seg_data_full = {'startX': start_x,
                         'startY': start_y,
                         'endX': end_x,
                         'endY': end_y,
                         'segNumbers': num_seg,
                         'offsetSeg': off_seg,
                         'Quality': quality,
                         'Shift': shift,
                         'SegLayers': b_scans_segments}
        return seg_data_full
