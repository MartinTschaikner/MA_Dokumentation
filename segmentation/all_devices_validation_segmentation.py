"""
Creates Scores and segmentation data for given validation data and chosen model.
"""

from keras.models import load_model
from prediction_stitching_to_seg_to_bin import PredictionStitchingToSegmentation
from split_resize_scan_to_windows import *
from losses import *
from keras.utils import np_utils
from validation_scores import *
from segmentation_countor_to_array import contour_to_array
from tkinter.filedialog import askdirectory
from os import walk

# window width, height for model input and stride of window over ring scan image
width = 128
height = 512
stride_x = 64
ringscan_width = 768
# resolutions in mum
res_all = [2000/1024, 2300/885]


save_path = 'SavedNetworks/'
results_save_dir = 'validation_transfer/'

dir_models = askdirectory(title='Choose directory of saved models to compute corresponding scores!')
(_, _, models) = next(walk(dir_models))

# load ring scans from all devices and labels, resize and concatenate for validation
rs_stack_S = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_stack.npy')[-10:, :, :, :]
rs_stack_C = np.load(r'data\validation_Cirrus_ring_scan\Cirrus_ring_scan_stack_f2.npy')[-10:, :, :, :]
rs_stack_T = np.load(r'data\validation_Topcon_ring_scan\Topcon_ring_scan_stack_f2.npy')[-10:, :, :, :]
# KEIN Spectralis !
ring_scan_all = [rs_stack_C, rs_stack_T]

y_test_S = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_label_stack.npy')[-10:, :, :, :]
y_test_C = np.load(r'data\validation_Cirrus_ring_scan\Cirrus_ring_scan_label_stack_f2.npy')[-10:, :, :, :]
y_test_T = np.load(r'data\validation_Topcon_ring_scan\Topcon_ring_scan_label_stack_f2.npy')[-10:, :, :, :]
Y_test_C = np_utils.to_categorical(y_test_C, num_classes=4, dtype=np.uint8)
Y_test_T = np_utils.to_categorical(y_test_T, num_classes=4, dtype=np.uint8)
# KEIN Spectralis !
y_test_all = [y_test_C, y_test_T]
Y_test_all = [Y_test_C, Y_test_T]


for m in range(len(models)):
    model_name = models[m]
    model_file_name = dir_models + '/' + models[m]
    model_dir = model_name[:-3]

    if model_dir.find('dice') != -1:
        custom = {'cce_dice_loss': cce_dice_loss}
    elif model_dir.find('tversky') != -1:
        custom = {'cce_tvesky_loss7': cce_tvesky_loss7}
    elif model_dir.find('jaccard') != -1:
        custom = {'cce_jaccard_loss': cce_jaccard_loss}
    else:
        custom = None

    # import model
    model = load_model(model_file_name, custom_objects=custom)

    # placeholder prediction labels
    Y_prediction_all = [np.empty(Y_test_C.shape).astype(np.uint8), np.empty(Y_test_T.shape).astype(np.uint8)]

    # placeholder scores (num_labels, num_scores, num_scans)
    scores_all = [np.empty((4, 8, np.size(rs_stack_C, 0))), np.empty((4, 8, np.size(rs_stack_T, 0)))]
    errors_all = [np.empty((3, 2, np.size(rs_stack_C, 0))), np.empty((3, 2, np.size(rs_stack_T, 0)))]

    # placeholder segmentation 2 channels ground truth and CNN segmentation
    final_segmentation_all = [np.empty((np.size(rs_stack_C, 0), 3, ringscan_width, 2)),
                              np.empty((np.size(rs_stack_T, 0), 3, ringscan_width, 2))]

    for j in range(len(ring_scan_all)):
        ring_scan_stack = ring_scan_all[j]
        y_test = y_test_all[j]
        Y_test = Y_test_all[j]
        res = res_all[j]
        for i in range(np.size(ring_scan_stack, 0)):
            # prepare model input data by resizing and splitting ring scan into overlapping windows
            ring_scan_window_stack = windowing(ring_scan_stack[i:i+1, :, :, :],
                                               'ring scan(s)', width, height, stride_x, 1)
            ring_scan_window_stack = ring_scan_window_stack/255
            num_windows_per_ring_scan = np.ceil(np.size(ring_scan_window_stack, 0)/np.size(ring_scan_stack, 0))\
                .astype('int')
            print(num_windows_per_ring_scan, 'windows per ring scan.')

            # compute model prediction for each window
            prediction_model = model.predict(ring_scan_window_stack, batch_size=1, verbose=1)

            # compute stitched prediction and results
            prediction_segmentation = PredictionStitchingToSegmentation(prediction_model, stride_x,
                                                                        np.size(ring_scan_stack, 1))
            stitched_prediction = prediction_segmentation.stitching_prediction(filter_type='gaussian')
            all_segmentation = prediction_segmentation.\
                segmentation_stitched_prediction(stitched_prediction)
            stitched_prediction_binary = prediction_segmentation.\
                seg_to_binary_prediction(all_segmentation, stitched_prediction)

            # compute ground truth segmentation
            ground_truth_segmentation = prediction_segmentation. \
                segmentation_stitched_prediction(y_test[i, :, :, 0])

            Y_prediction_all[j][i, :, :, :] = stitched_prediction_binary

            # plot segmentation
            # prediction_segmentation.plot_stitched_prediction(stitched_prediction)
            # prediction_segmentation.plot_binaries(stitched_prediction_binary)
            # prediction_segmentation.plot_segmentation(ring_scan_stack[i:i+1, :, :, :],
                                                  # all_segmentation,
                                                  # ground_truth_segmentation)

            # transform contour to array of correct length
            segmentation = contour_to_array(all_segmentation, np.size(ring_scan_stack, 2))
            ground_truth = contour_to_array(ground_truth_segmentation, np.size(ring_scan_stack, 2))

            final_segmentation_all[j][i, :, :, 0] = segmentation
            final_segmentation_all[j][i, :, :, 1] = ground_truth

            errors_all[j][:, 0, i], errors_all[j][:, 1, i] = un_signed_error(segmentation, ground_truth)

            for k in range(np.size(Y_test, 3)):
                scores_all[j][k, 0, i] = dice_sc(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])
                scores_all[j][k, 1, i] = accuracy(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])
                scores_all[j][k, 2, i] = precision(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])
                scores_all[j][k, 3, i] = specificity(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])
                scores_all[j][k, 4, i] = sensitivity(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])
                scores_all[j][k, 5, i] = balanced_accuracy(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])
                scores_all[j][k, 6, i] = jaccard(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])
                scores_all[j][k, 7, i] = tversky7(Y_test[i, :, :, k], Y_prediction_all[j][i, :, :, k])

    scores = np.concatenate((scores_all[0], scores_all[1]), axis=2)
    errors = np.concatenate((errors_all[0]*res_all[0], errors_all[1]*res_all[1]), axis=2)
    final_segmentation = np.concatenate((final_segmentation_all[0], final_segmentation_all[1]), axis=0)

    np.save(results_save_dir + model_dir + '/' + 'scores.npy', scores)
    np.save(results_save_dir + model_dir + '/' + 'errors.npy', errors)
    np.save(results_save_dir + model_dir + '/' + 'segmentation.npy', final_segmentation)
    np.save(results_save_dir + model_dir + '/' + 'ring_scans_C.npy', rs_stack_C)
    np.save(results_save_dir + model_dir + '/' + 'ring_scans_T.npy', rs_stack_T)
