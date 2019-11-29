from keras.models import load_model
from prediction_stitching_to_seg_to_bin import PredictionStitchingToSegmentation
from split_resize_scan_to_windows import *
from losses import *
from validation_scores import *
from segmentation_countor_to_array import contour_to_array

# window width, height for model input and stride of window over ring scan image
width = 128
height = 512
stride_x = 64
ringscan_width = 768

# load ring scans from all devices, add one dim for CNN
rs_stack_S = np.load(r'data\Spectralis_val_ring_scans_f1.npy')
rs_stack_S = np.moveaxis(rs_stack_S, -1, 0).astype(np.uint8)

rs_stack_S = np.expand_dims(rs_stack_S, axis=3)

rs_stack_C = np.load(r'data\Cirrus_validation_ring_scan_stack_f2.npy')
rs_stack_C = np.expand_dims(rs_stack_C, axis=3)

rs_stack_T = np.load(r'data\Topcon_validation_ring_scan_stack_f2.npy')
rs_stack_T = np.expand_dims(rs_stack_T, axis=3)

stacks = [rs_stack_S, rs_stack_C, rs_stack_T]


# import model
model = load_model('model/model_k5_c8_xe_tversky7_dout_transfer_final.h5',
                   custom_objects={'cce_tvesky_loss7': cce_tvesky_loss7})

# placeholder all segmentations for 3 devices x 20 scans x 3 segmentation lines x 768 length
final_segmentation = np.empty((3, np.size(rs_stack_S, 0), 3, np.size(rs_stack_S, 2)))

for j in range(len(stacks)):
    ring_scan_stack = stacks[j]

    for i in range(np.size(ring_scan_stack, 0)):

        # prepare model input data by resizing and splitting ring scan into overlapping windows
        ring_scan_window_stack = windowing(ring_scan_stack[i:i+1, :, :, :], 'ring scan(s)', width, height, stride_x, 1)
        ring_scan_window_stack = ring_scan_window_stack/255
        num_windows_per_ring_scan = np.size(ring_scan_window_stack, 0)
        print(num_windows_per_ring_scan, 'windows per ring scan.')

        # compute model prediction for each window
        prediction_model = model.predict(ring_scan_window_stack, batch_size=1, verbose=1)

        # compute stitched prediction and results
        prediction_segmentation = PredictionStitchingToSegmentation(prediction_model,
                                                                    stride_x, np.size(ring_scan_stack, 1))
        stitched_prediction = prediction_segmentation.stitching_prediction(filter_type='gaussian')

        # plot segmentation
        # prediction_segmentation.plot_stitched_prediction(stitched_prediction)

        all_segmentation = prediction_segmentation.\
            segmentation_stitched_prediction(stitched_prediction)

        # transform contour to array of correct length
        segmentation = contour_to_array(all_segmentation, np.size(ring_scan_stack, 2))
        final_segmentation[j, i, :, :] = segmentation

np.save('segmentation_validation_data/final_segmentation.npy', final_segmentation)
