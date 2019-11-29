"""
A class to stitch predictions of image stripes together to gain a prediction of the whole input image.
Then calculate categorical binary masks and deduce segmentation from the morphological closed masks.

@author = Martin Tschaikner

@Date = 13.09.2019
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.transform import resize


class PredictionStitchingToSegmentation:
    """
    This class computes a segmentation from a STITCHED CNN label prediction stack of an input image.
    """

    def __init__(self, prediction, stride_x, ring_scan_height):

        self.prediction = prediction
        self.stride_x = stride_x
        self.ring_scan_height = ring_scan_height

    def stitching_prediction(self, filter_type):

        """
        This method stitches the predictions for each input windows to a prediction for the original ring scan.

        :param filter_type: weight single predictions (less weight at prediction boarders, more in center)
        :return: stitched prediction
        :rtype:
        """
        # width image
        width = np.size(self.prediction, 2)
        num_windows = np.size(self.prediction, 0)
        total_width = (num_windows-1) * self.stride_x + width

        # placeholder prediction whole image
        total_prediction = np.empty((num_windows, np.size(self.prediction, 1),
                                     total_width, np.size(self.prediction, 3)))

        if filter_type == 'gaussian':
            mu = (width - 1)/2
            sigma = width/8
            weights = np.array(np.exp(-np.square(np.arange(width) - mu) / 2 / np.square(sigma))).reshape(1, -1)
            weights = np.expand_dims(weights, axis=3)
        elif filter_type == 'rectangular':
            weights = np.zeros((1, width))
            a_rect = width/4
            weights[:, int(width/2-a_rect):int(width/2+a_rect)] = 1
            weights = np.expand_dims(weights, axis=3)
        else:
            weights = 1

        for i in range(num_windows):
            total_prediction[i, :, i * self.stride_x:i * self.stride_x + width, :] = \
                self.prediction[i, :, :, :] * weights

        if filter_type == 'rectangular':
            a_rect = width/4
            total_prediction[0, :, :int(a_rect), :] = \
                self.prediction[0, :, :int(a_rect), :]
            total_prediction[-1, :, -int(a_rect):, :] = \
                self.prediction[-1, :, -int(a_rect):, :]

        total_prediction = np.sum(total_prediction, axis=0)

        # find label with maximum probability and create placeholder for categorical binaries
        total_prediction_resize = resize(total_prediction,
                                         (self.ring_scan_height, total_width), anti_aliasing=True, mode='reflect')
        stitched_prediction = np.argmax(total_prediction_resize, axis=2)

        return stitched_prediction

    @staticmethod
    def segmentation_stitched_prediction(stitched_prediction):

        """
        This method transforms the stitched prediction to binary predictions for each categorical label.
        Then morphological filters are used to close holes and compute contours to delete holes at image boarders
        and to exclude wrong segmentation.

        :param stitched_prediction:
        :return: segmentation list
        """

        # initialize segmentation list for segmentation of all predictions
        final_segmentation = []
        all_segmentation = []

        levels = [0.99, 1.99, 2.99]
        for i in range(3):
            contours = measure.find_contours(stitched_prediction, levels[i])

            # loop over all detected contours and exclude impossible contours
            ind_contours = np.zeros(np.size(contours, 0))

            for n, contour in enumerate(contours):
                contour = np.around(contour).astype('int')
                touch_left = np.argwhere(contour[:, 1] == 0).size
                touch_right = np.argwhere(contour[:, 1] == np.size(stitched_prediction, 1) - 1).size
                touch_top = np.argwhere(contour[:, 0] == 0).size

                # define condition for possible segmentation, excluding holes at left, right and bottom image margin
                if touch_left != 0 and touch_right != 0 or touch_left != 0 and touch_top != 0 or \
                        touch_right != 0 and touch_top != 0:
                    ind_contours[n] = 1

            # only keep plausible segmentation
            ind_seg = np.argwhere(ind_contours == 1)[:, 0]
            segmentation = [contours[i].astype('int') for i in ind_seg]

            # extend segmentation list
            all_segmentation.append(segmentation)

        # pick contours for final segmentation
        final_segmentation.append(all_segmentation[0][0])
        if len(all_segmentation[1]) == 0 and len(all_segmentation[2]) == 0:
            final_segmentation.append(all_segmentation[0][0])
        else:
            final_segmentation.append(all_segmentation[1][0])

        final_segmentation.append(all_segmentation[2][0])

        if np.all(final_segmentation[0] == final_segmentation[2]):
            final_segmentation[0] = all_segmentation[0][2]
            final_segmentation[1] = all_segmentation[1][2]
            final_segmentation[2] = all_segmentation[2][2]

        return final_segmentation

    def seg_to_binary_prediction(self, final_segmentation, stitched_prediction):

        num_classes = np.size(self.prediction, 3)
        ring_scan_width = np.size(stitched_prediction, 1)
        stitched_prediction_binary = np.zeros((np.size(stitched_prediction, 0),
                                               ring_scan_width,
                                               num_classes)).astype(np.uint8)

        all_segmentation = np.empty((np.size(final_segmentation, 0), np.size(stitched_prediction, 1)))
        all_segmentation[:] = np.nan
        for n, segmentation in enumerate(final_segmentation):
            seg_x = np.around(segmentation[:, 1]).astype('int')
            u, indices = np.unique(seg_x, axis=0, return_index=True)
            all_segmentation[n, seg_x[indices]] = segmentation[indices, 0]

        all_segmentation = all_segmentation.astype('int')

        for i in range(num_classes):

            for j in range(ring_scan_width):
                if i == 0:
                    stitched_prediction_binary[:all_segmentation[i, j], j, i] = 1
                elif i == num_classes - 1:
                    stitched_prediction_binary[all_segmentation[i - 1, j]:, j, i] = 1
                else:
                    stitched_prediction_binary[all_segmentation[i - 1, j]:all_segmentation[i, j], j, i] = 1

        return stitched_prediction_binary

    @staticmethod
    def plot_binaries(stitched_prediction_binary):

        # plot parameter
        noe = np.size(stitched_prediction_binary, 1)
        num_rows = np.size(stitched_prediction_binary, 0)

        fig, ax = plt.subplots(nrows=2, ncols=2)

        fig.suptitle('Categorical labels', fontsize=25)

        ax[0, 0].imshow(stitched_prediction_binary[:, :, 0], cmap='gray', vmin=0, vmax=1, extent=(0, noe, num_rows, 0))
        ax[0, 0].set_title('Label 0', pad=22)
        ax[0, 0].title.set_size(25)
        ax[0, 0].set_xlabel('number of A scans [ ]', labelpad=18)
        ax[0, 0].set_ylabel('Z axis [ ]', labelpad=18)
        ax[0, 0].xaxis.label.set_size(20)
        ax[0, 0].yaxis.label.set_size(20)

        ax[0, 1].imshow(stitched_prediction_binary[:, :, 1], cmap='gray', vmin=0, vmax=1, extent=(0, noe, num_rows, 0))
        ax[0, 1].set_title('Label 1', pad=22)
        ax[0, 1].title.set_size(25)
        ax[0, 1].set_xlabel('number of A scans [ ]', labelpad=18)
        ax[0, 1].set_ylabel('Z axis [ ]', labelpad=18)
        ax[0, 1].xaxis.label.set_size(20)
        ax[0, 1].yaxis.label.set_size(20)

        ax[1, 0].imshow(stitched_prediction_binary[:, :, 2], cmap='gray', vmin=0, vmax=1, extent=(0, noe, num_rows, 0))
        ax[1, 0].set_title('Label 2', pad=22)
        ax[1, 0].title.set_size(25)
        ax[1, 0].set_xlabel('number of A scans [ ]', labelpad=18)
        ax[1, 0].set_ylabel('Z axis [ ]', labelpad=18)
        ax[1, 0].xaxis.label.set_size(20)
        ax[1, 0].yaxis.label.set_size(20)

        ax[1, 1].imshow(stitched_prediction_binary[:, :, 3], cmap='gray', vmin=0, vmax=1, extent=(0, noe, num_rows, 0))
        ax[1, 1].set_title('Label 3', pad=22)
        ax[1, 1].title.set_size(25)
        ax[1, 1].set_xlabel('number of A scans [ ]', labelpad=18)
        ax[1, 1].set_ylabel('Z axis [ ]', labelpad=18)
        ax[1, 1].xaxis.label.set_size(20)
        ax[1, 1].yaxis.label.set_size(20)

        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.subplots_adjust(top=0.9)

        plt.show()

    @staticmethod
    def plot_stitched_prediction(stitched_prediction):

        # plot parameter
        noe = np.size(stitched_prediction, 1)
        num_rows = np.size(stitched_prediction, 0)

        fig, ax = plt.subplots(nrows=1)

        ax.imshow(stitched_prediction[:, :], cmap='gray', vmin=0, vmax=3, extent=(-0.5, noe - 0.5, num_rows, 0))
        ax.set_title('Resized stitched prediction', pad=22)
        ax.title.set_size(25)
        ax.set_xlabel('number of A scans [ ]', labelpad=18)
        ax.set_ylabel('Z axis [ ]', labelpad=18)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)

        plt.show()

    @staticmethod
    def plot_segmentation(ring_scan_stack, all_segmentation, ground_truth_seg):

        # plot stitched predictions
        noe = np.size(ring_scan_stack, 2)
        num_rows = np.size(ring_scan_stack, 1)

        fig, ax0 = plt.subplots(ncols=1)

        if len(ground_truth_seg) != 0:
            for n, segmentation in enumerate(ground_truth_seg):
                if n == 0:
                    ax0.plot(segmentation[:, 1], segmentation[:, 0], linewidth=2, color='yellow', label='gold standard')
                ax0.plot(segmentation[:, 1], segmentation[:, 0], linewidth=2, color='yellow')

        for n, segmentation in enumerate(all_segmentation):
            # color = ['green', 'green', 'blue', 'blue', 'red', 'red']
            if n == 0:
                ax0.plot(segmentation[:, 1], segmentation[:, 0], linewidth=1, color='blue', label='segmentation')
            ax0.plot(segmentation[:, 1], segmentation[:, 0], linewidth=1, color='blue')

        ax0.imshow(ring_scan_stack[0, :, :, 0], cmap='gray', vmin=0, vmax=255, extent=(-1, noe, num_rows, 0))
        ax0.set_title('Ring scan with segmentation', pad=22)
        ax0.title.set_size(25)
        ax0.set_xlabel('number of A scans [ ]', labelpad=18)
        ax0.set_ylabel('Z axis [ ]', labelpad=18)
        ax0.xaxis.label.set_size(20)
        ax0.yaxis.label.set_size(20)
        ax0.legend(loc='lower right', shadow=True, fontsize='x-large')

        plt.show()
