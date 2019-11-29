"""
This function transforms n 2D contours of form (2, contour len) into a 2D array of form (n, ring scan width)
for further use in OCT marker export and comparisons between absolute pixel error of segmentation.
"""

import numpy as np


def contour_to_array(contour, ring_scan_width):

    # placeholder for segmentation (delete doubles in x direction)
    all_segmentation = np.empty((np.size(contour, 0), ring_scan_width))
    all_segmentation[:] = np.nan
    for n, segmentation in enumerate(contour):
        seg_x = np.around(segmentation[:, 1]).astype('int')
        u, indices = np.unique(seg_x, axis=0, return_index=True)
        all_segmentation[n, seg_x[indices]] = segmentation[indices, 0]

    json_all_segmentation = all_segmentation[0::2, :]

    if np.size(all_segmentation, 0) < 4:
        json_all_segmentation = all_segmentation

    return json_all_segmentation
