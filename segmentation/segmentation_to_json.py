"""
This script segments a stack of ring scans and outputs the segmentation as csv.

"""
from keras.models import load_model
from prediction_stitching_to_seg_to_bin import PredictionStitchingToSegmentation
from split_resize_scan_to_windows import windowing
from jason_marker_array_converter import array_to_json
from losses import *
from segmentation_countor_to_array import contour_to_array
import cv2
from random import randrange
import os
import numpy as np

device = 'Cirrus'
filter_parameter = 2

# output path for json and tiff, path of container (dummy)
dummy_path = r'marker_dummy\marker_dummy.joctmarker'
tempDir = device + '_temp_' + str(randrange(1000)) + '/'
tiff_path = r'interpolated_ring_scan_marker_tiff/' + tempDir
os.mkdir(tiff_path)

# image width and height for model input
width = 128
height = 512

# import model
save_path = 'SavedNetworks'
model = load_model(save_path + '/model_k5_c8_xe_tversky7_dout_transfer_final.h5',
                   custom_objects={'cce_tvesky_loss7': cce_tvesky_loss7})

# import interpolated ring scan (stack)
ring_scan_stack = np.load(r'data\\' + device + r'_ring_scan_int'
                          r'\\' + device + '_ring_scan_stack_f' + str(filter_parameter) + '.npy')

# input size for further computation expects 4d array
if len(ring_scan_stack.shape) == int(2):
    ring_scan_stack = np.expand_dims(ring_scan_stack, axis=0)
if len(ring_scan_stack.shape) == int(3):
    ring_scan_stack = np.expand_dims(ring_scan_stack, axis=3)

# resize and split stack into overlapping windows and normalize
stride_x = 64

for i in range(np.size(ring_scan_stack, 0)):
    # prepare model input data by resizing and splitting ring scan into overlapping windows
    ring_scan_window_stack = windowing(ring_scan_stack[i:i+1, :, :, :], 'ring scan(s)', width, height, stride_x, 1)
    ring_scan_window_stack = ring_scan_window_stack/255

    # compute model prediction for each window
    prediction_model = model.predict(ring_scan_window_stack, batch_size=1, verbose=1)

    # compute stitched prediction and results
    prediction_segmentation = PredictionStitchingToSegmentation(prediction_model, stride_x, np.size(ring_scan_stack, 1))
    stitched_prediction = prediction_segmentation.stitching_prediction(filter_type='gaussian')
    all_segmentation = prediction_segmentation.\
        segmentation_stitched_prediction(stitched_prediction)
    stitched_prediction_binary = prediction_segmentation.seg_to_binary_prediction(all_segmentation, stitched_prediction)
    print(len(all_segmentation))

    # plot segmentation
    # prediction_segmentation.plot_stitched_prediction(stitched_prediction)
    # prediction_segmentation.plot_binaries(stitched_prediction_binary)
    # prediction_segmentation.plot_segmentation(ring_scan_stack[i:i+1, :, :, :], all_segmentation, all_segmentation)

    # convert contour to array (3, ring scan width)
    json_all_segmentation = contour_to_array(all_segmentation, np.size(ring_scan_stack, 2))
    print(np.size(json_all_segmentation, 1))

    # save ring scan as tiff for import in OCT marker
    file_tiff_name = device + '_ring_scan_f' + str(filter_parameter) + '_' + str(i) + '_int.tiff'
    cv2.imwrite(tiff_path + file_tiff_name, ring_scan_stack[i, :, :, 0])

    # saves segmentation as json oct marker file
    new_file_name = tiff_path + file_tiff_name + '.joctmarker'
    array_to_json(json_all_segmentation, dummy_path, new_file_name)
