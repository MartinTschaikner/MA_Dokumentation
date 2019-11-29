"""
A class to interpolate a ring scan with given radius and center (BMO points) from a given volume scan.

@author = Martin Tschaikner

@Date = 07.07.2019
"""

# import numerical python, statistics and plot packages
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt


class RingScanFromVolume:
    """
    This class computes interpolated ring scan data (image, ilm and rpe segmentation) from a given volume scan.
    """

    def __init__(self, file_header, b_scan_stack, seg_data_full, file_name_bmo, radius, number_circle_points,
                 filter_parameter):

        """
        Declaration of the class variables.

        :param file_header: header of vol file
        :type file_header: dictionary
        :param b_scan_stack: b scan data of vol file
        :type b_scan_stack: dictionary
        :param seg_data_full: segmentation data of vol file
        :type seg_data_full: dictionary
        :param file_name_bmo: text file of bmo points of corresponding vol file
        :type file_name_bmo: csv text file
        :param radius: radius of ring scan
        :type radius: float - default: 1.75 mm
        :param number_circle_points: number of equidistant circle points on ring scan
        :type number_circle_points: integer
        :param filter_parameter: weighting parameter for ring scan interpolation
        :type filter_parameter: integer

        """

        self.file_header = file_header
        self.b_scan_stack = b_scan_stack
        self.seg_data_full = seg_data_full
        self.file_name_bmo = file_name_bmo
        self.radius = radius
        self.number_circle_points = number_circle_points
        self.filter_parameter = filter_parameter

    def circle_points_coordinates(self):

        """
        Method to compute circle points for ring scan interpolation, center of circle defined as geometric mean
        of BMO points

        :return: circle points for interpolation
        :rtype: 2d float array - 2 x number of circle points
        """

        # bmo correction
        bmo_crop = 15
        # import bmo points, scaling, computing of bmo center as mean of all points and projection on x,y plane
        bmo_data = pd.read_csv(self.file_name_bmo, sep=",", header=None)
        bmo_points = np.zeros([bmo_data.shape[1], bmo_data.shape[0]], dtype=float)
        bmo_points[:, 0] = bmo_data.iloc[0, :] * self.file_header['Distance']
        bmo_points[:, 1] = (bmo_data.iloc[1, :] + bmo_crop) * self.file_header['ScaleX']
        bmo_points[:, 2] = bmo_data.iloc[2, :] * self.file_header['ScaleZ']
        bmo_center_3d = np.mean(bmo_points, axis=0)
        bmo_center_2d = np.array([bmo_center_3d[0], bmo_center_3d[1]])
        bmo_center_geom_mean_2d = geometric_median(bmo_points[:, 0:2], eps=1e-6)

        # compute noe equidistant circle points around bmo center with given radius
        noe = self.number_circle_points

        # Scan Position
        scan_pos = str(self.file_header['ScanPosition'])
        scan_pos_input = scan_pos[2:4]

        # OD clock wise, OS ccw ring scan interpolation for correct orientation
        if scan_pos_input == 'OS':
            phi = np.linspace(0, - 2 * np.pi, num=noe, endpoint=False)
        else:
            phi = np.linspace(0, 2 * np.pi, num=noe, endpoint=False) - np.pi

        # create center from geometric median as vector for broadcasting
        center = np.linspace(bmo_center_geom_mean_2d, bmo_center_geom_mean_2d, num=noe).T

        # compute circle points with given center and radius
        circle_points_coordinates = center + self.radius * np.array((np.cos(phi), np.sin(phi)))

        # plot to visualize differences between center of mass vs. geom median
        plot = False
        if plot:
            radius_bmo = np.mean(np.sqrt((bmo_points[:, 0] - bmo_center_geom_mean_2d[0])**2 +
                                         (bmo_points[:, 1] - bmo_center_geom_mean_2d[1])**2))
            bmo_circle_points = center + radius_bmo * np.array((np.cos(phi), np.sin(phi)))
            shift_centers = np.sqrt((bmo_center_2d[0] - bmo_center_geom_mean_2d[0])**2 +
                                    (bmo_center_2d[1] - bmo_center_geom_mean_2d[1])**2)
            print('Difference between center of mass and geometric median of BMO points:', shift_centers)
            fig, ax = plt.subplots(ncols=1)
            ax.plot(bmo_points[:, 0], bmo_points[:, 1], color='black', marker='o', linestyle='None')
            ax.plot(bmo_center_3d[0], bmo_center_3d[1], color='red', marker='o', linestyle='None')
            ax.plot(bmo_center_geom_mean_2d[0], bmo_center_geom_mean_2d[1], color='orange', marker='+')
            ax.plot(bmo_circle_points[0], bmo_circle_points[1], color='black', linestyle='None')
            ax.plot(circle_points_coordinates[0], circle_points_coordinates[1], ':', linewidth=3, color='green')
            ax.set_aspect('equal', 'box')
            ax.axis('off')
            plt.show()

        return circle_points_coordinates

    def ring_scan_interpolation(self, circle_points_coordinates):

        """
        This method computes the interpolation. Therefor gaussian weights for those A scans within a filter mask
        surrounding a given circle point are computed and a new interpolated value is created with those weights.

        :param circle_points_coordinates: circle points for interpolation
        :type circle_points_coordinates: 2d float array - 2 x number of circle points
        :return: interpolated grey value image, smoothed ilm & rpe segmentation from Heyex and boolean
                 if segmentation was successful
        :rtype: 2d float array, 2x 1d float array boolean
        """

        noe = self.number_circle_points

        # create data arrays for interpolation
        ring_scan_data = np.zeros([self.file_header['SizeZ'], noe])
        ilm_ring_scan = np.zeros(noe)
        rpe_ring_scan = np.zeros(noe)

        # check number of segmentation layers within volume file
        for i in range(np.size(self.seg_data_full['SegLayers'], 1)):
            layer = self.seg_data_full['SegLayers'][:, i, :]
            # if complete layer is nan, exit loop and return number of layers
            if np.nansum(layer) == 0:
                print('Number of segmentation layers:', i)
                break
            if i == np.size(self.seg_data_full['SegLayers'], 1) - 1:
                print('Number of segmentation layers:', str(i + 1))

        # filter size computed depending on ratio of x to y resolution of volume data
        if self.file_header['Distance'] > self.file_header['ScaleX']:
            multiples = np.around(self.file_header['Distance'] / self.file_header['ScaleX'])
            f_x = int(2 * self.filter_parameter + 1)
            f_y = int(2 * multiples * self.filter_parameter + 1)
            # defines sigma as 1/3 of greater distance
            sigma = 1 / 3 * max(self.file_header['Distance'], multiples * self.file_header['ScaleX'])
        else:
            multiples = np.around(self.file_header['ScaleX'] / self.file_header['Distance'])
            f_x = int(2 * multiples * self.filter_parameter + 1)
            f_y = int(2 * self.filter_parameter + 1)
            # defines sigma as 1/3 of greater distance
            sigma = 1 / 3 * max(multiples * self.file_header['Distance'], self.file_header['ScaleX'])

        # sigma^2
        if int(self.filter_parameter) != 0:
            sigma2 = np.square(self.filter_parameter * sigma)
        else:
            # defines sigma as 1/3 of smaller distance
            sigma2 = np.square(1 / 3 * min(self.file_header['Distance'], self.file_header['ScaleX']))

        # loop over all circle points to gain interpolated values
        for i in range(noe):
            # reshape and calculating indices of nearest data grid point to ith circle point
            loc_var = np.reshape(circle_points_coordinates[:, i], [2, 1])
            index_x_0 = np.around(loc_var[0] / self.file_header['Distance']).astype(int)
            index_y_0 = np.around(loc_var[1] / self.file_header['ScaleX']).astype(int)

            # compute range of indices for filter mask
            index_x_min = int(index_x_0 - (f_x - 1) / 2)
            index_x_max = int(index_x_0 + (f_x - 1) / 2)
            index_y_min = int(index_y_0 - (f_y - 1) / 2)
            index_y_max = int(index_y_0 + (f_y - 1) / 2)

            # fill filter mask with corresponding indices
            index_xx, index_yy = np.meshgrid(np.linspace(index_x_min, index_x_max, f_x),
                                             np.linspace(index_y_min, index_y_max, f_y))

            # compute matrix of indices differences in x, y direction
            diff_x = index_xx - loc_var[0] / self.file_header['Distance']
            diff_y = index_yy - loc_var[1] / self.file_header['ScaleX']

            # compute weights and interpolated grey values
            w = np.exp(-(np.square(diff_x * self.file_header['Distance']) +
                         np.square(diff_y * self.file_header['ScaleX'])) / (2 * sigma2))

            # get gray values within filter mask from volume data
            gv = self.b_scan_stack[:, index_y_min:index_y_max + 1, index_x_min:index_x_max + 1]

            # apply weights at grey values and compute interpolated value
            gv_w = np.sum(np.sum(w * gv, axis=1), axis=1) / np.sum(w)

            # set grey values greater 260 to 0
            gv_w[gv_w >= 0.260e3] = 0

            # fill ring scan data array
            ring_scan_data[:, i] = gv_w

            # repeat interpolation for ilm data array -- SegLayers # 0
            z_ilm = self.seg_data_full['SegLayers'][index_y_min:index_y_max + 1, 0, index_x_min:index_x_max + 1]

            # handle nan in ilm data with nan sum
            check = np.isnan(z_ilm).astype('int')
            if np.sum(check) != 0:
                # search for indices with nan entries and set corresponding weights to 0
                ind = np.where(check == int(1))
                w[ind] = 0
                # print("nan in ILM data for circle point #", i, "@ index center", index_x_0, index_y_0)

            # interpolate data
            ilm_ring_scan[i] = np.nansum(w * z_ilm) / np.sum(w)

            # repeat for rpe data array -- SegLayers # 1
            z_rpe = self.seg_data_full['SegLayers'][index_y_min:index_y_max + 1, 1, index_x_min:index_x_max + 1]

            # handle nan in rpe data with nan sum
            check = np.isnan(z_rpe).astype('int')
            if np.sum(check) != 0:
                if np.sum(check) != int(f_x * f_y):
                    # search for indices with nan entries and set corresponding weights to 0
                    ind = np.where(check == int(1))
                    w[ind] = 0
                    # interpolate data
                    rpe_ring_scan[i] = np.nansum(w * z_rpe) / np.sum(w)
                    # print("nan in RPE data for circle point #", i, "@ index center", index_x_0, index_y_0)

                # if all entries are nan, linear interpolate from nearest neighbor points
                else:
                    rpe_ring_scan[i] = 2 * rpe_ring_scan[i - 1] - rpe_ring_scan[i - 2]
            else:
                # interpolate data
                rpe_ring_scan[i] = np.sum(w * z_rpe) / np.sum(w)

        # smooth interpolated rpe and ilm segmentation and decide if data is used or not
        rpe_ring_scan, remove_boolean_rpe = smooth_segmentation(rpe_ring_scan, 300)
        if remove_boolean_rpe is True:
            print('\x1b[0;30;41m', 'Smoothing failed (probably poor rpe segmentation of volume scan) : data '
                  'not used for Bland-Altman plot!', '\x1b[0m')

        ilm_ring_scan, remove_boolean_ilm = smooth_segmentation(ilm_ring_scan, 100)
        if remove_boolean_ilm is True:
            print('\x1b[0;30;41m', 'Smoothing failed (probably poor ilm segmentation of volume scan) : data '
                  'not used for Bland-Altman plot!', '\x1b[0m')

        remove_boolean = remove_boolean_rpe or remove_boolean_ilm

        return ring_scan_data, ilm_ring_scan, rpe_ring_scan, remove_boolean


def smooth_segmentation(segmentation_data, smoothing_factor):
    """
    This static method checks the interpolated segmentation for critical points (gradient) and if there are any
    deletes them and fits a 3rd degree order spline through the remaining segmentation points. The smoothed segmentation
    is then checked once more and if there still exist critical points, the segmentation smoothing has failed and
    the data will not be compared to an actual ring scan

    :param segmentation_data: interpolated ring scan segmentation data (ilm or rpe)
    :type segmentation_data: 1-dim float array
    :param smoothing_factor: smoothing factor of spline fitting (around 100 to 300)
    :type: float
    :return: smoothed segmentation spline, boolean for smoothing ok/failed
    :rtype: 1-dim float array, boolean
    """

    noe = np.size(segmentation_data, 0)

    # line space for spline fitting
    x_spline = np.linspace(0, noe - 1, noe)

    # number of neighbors counted as critical points as well
    num_neighbor = 4

    # compute critical indices plus neighbours
    critical_indices = critical_points(segmentation_data, num_neighbor, 3.5)

    # minimal length of connected non critical part
    critical_length = 20

    # splits up the connected non critical parts of the segmentation and adds a part to critical if it´s length is
    # smaller than the critical length
    final_indices = segmentation_critical_split(segmentation_data, critical_indices, critical_length).astype('int')

    # resulting line space for spline fit
    x_fit = np.delete(x_spline, final_indices).astype('int')

    # compute 3 deg order spline fit
    spl = UnivariateSpline(x_fit, segmentation_data[x_fit], s=int(noe))
    spl.set_smoothing_factor(smoothing_factor)
    y_spline = spl(x_spline)

    # plot spline fit and compare to original segmentation
    plot = False
    if plot:
        plt.rcParams.update({'font.size': 18})
        plt.figure()
        plt.plot(x_fit, segmentation_data[x_fit], 'ro', ms=4)
        if critical_indices.size != 0:
            plt.plot(critical_indices, segmentation_data[critical_indices], 'bx', ms=6)
            plt.plot(final_indices, segmentation_data[final_indices], 'b+', ms=6)
        plt.plot(np.linspace(0, noe - 1, noe), segmentation_data, 'b', lw=1, label='interpolated segmentation')
        plt.plot(x_spline, y_spline, 'g', lw=3, label='3rd degree spline')
        plt.ylim([0, 300])
        plt.legend(loc='best')
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.show()

    # checks for spikes in spline fit and if there are any, exclude from statistics.
    std_factor = 8
    spline_critical = critical_points(y_spline, 0, std_factor).astype('int')

    if np.size(spline_critical):
        remove_boolean = True
    else:
        remove_boolean = False

    return y_spline, remove_boolean


def critical_points(segmentation_data, num_neighbor, factor_std):
    """
    This static method computes critical points, defined as points which gradients are off the mean gradient (of all
    segmentation points) for more than the standard deviation times a chosen factor

    :param segmentation_data: interpolated ring scan segmentation data (ilm or rpe)
    :type segmentation_data: 1-dim float array
    :param num_neighbor: number of neighbors counted as critical points as well
    :type num_neighbor: integer
    :param factor_std: regulates when a segmentation data point is flagged as critical
    :type factor_std: float
    :return: indices of critical points
    :rtype: 1-dim integer array
    """

    # compute number of ilm/rpe data points and corresponding gradient, mean and std of grad
    grad_data = np.gradient(segmentation_data)
    mean_grad = np.mean(grad_data)
    std_grad = np.std(grad_data)

    # compute critical points for which the gradient is far off (peaks)
    critical_indices = np.argwhere((grad_data > mean_grad + factor_std * std_grad) |
                                   (grad_data < mean_grad - factor_std * std_grad))

    # compute neighbourhood of critical points to close gaps, e.g. at max/min
    if critical_indices.size != 0:
        neighbor = np.linspace(-num_neighbor, num_neighbor, 2 * num_neighbor + 1).reshape((1, -1))
        critical_neighborhood = (critical_indices + neighbor).reshape((-1, 1))
        critical_indices = np.unique(critical_neighborhood, axis=0).astype('int')
    else:
        critical_indices = np.array([[]])

    return critical_indices


def segmentation_critical_split(segmentation_data, critical_indices, critical_length):
    """
    This static method flags connected non critical points as critical, if the length is smaller than a chosen critical
    length.

    :param segmentation_data: interpolated ring scan segmentation data (ilm or rpe)
    :type segmentation_data: 1-dim float array
    :param critical_indices: indices of critical points from corresponding method
    :type critical_indices: 1-dim integer array
    :param critical_length: minimal number of connected non critical points to not be flagged as critical
    :type critical_length: integer
    :return: final indices of critical points
    :rtype: 1-dim integer array
    """

    # checks if there are any critical points at all
    if critical_indices.size != 0:
        noe = np.size(segmentation_data, 0)
        final_indices = critical_indices.reshape(1, -1)

        # group_indices for finding start and end indices of non critical point groups
        group_indices = np.zeros(noe)
        group_indices[critical_indices] = int(1)
        group_indices[0] = int(1)
        group_indices = np.diff(group_indices).astype('int')

        # compute start/end indices of non critical point groups
        start_ind = (np.argwhere(group_indices == - int(1)) + 1).reshape(1, -1)
        start_ind[0, 0] = 0
        end_ind = np.argwhere(group_indices == int(1)).reshape(1, -1)
        if np.size(start_ind, 1) != np.size(end_ind, 1):
            end_ind = np.concatenate((end_ind, np.array([[noe - 1]])), axis=1)

        # compute length of non critical point groups
        group_len = (end_ind - start_ind).flatten()

        # compute indices for which group lengths are bigger than critical length
        length_pass = np.argwhere(group_len >= critical_length)

        # compute indices for non critical point groups smaller than the critical length
        start_ind_close = np.delete(start_ind, length_pass)
        end_ind_close = np.delete(end_ind, length_pass)

        # flags points within those groups as critical points
        if start_ind_close.size != 0:
            for i in range(len(start_ind_close)):
                new_critical = np.arange(start_ind_close[i], end_ind_close[i] + 1).reshape(1, -1)
                final_indices = np.concatenate((final_indices.reshape(1, -1), new_critical), axis=1)
            final_indices = np.unique(final_indices, axis=0).astype('int')
            final_indices = np.sort(final_indices)
    else:
        final_indices = critical_indices

    return final_indices


def geometric_median(bmo_points, eps=1e-5):
    """
    This static method computes the geometric median of the input coordinates
    (The multivariateL1-median and associated data depth Yehuda Vardi† and Cun-Hui Zhang‡ Department of Statistics,
    Rutgers University, New Brunswick, NJ 08854Communicated by Lawrence A. Shepp, Rutgers,
    The State University of New Jersey, Piscataway, NJ, November 17, 1999 (received for reviewOctober 15, 1999)

    :param bmo_points: BMO points coordinates
    :type bmo_points: 2d float array
    :param eps: parameter for accuracy of computation of geometric median
    :type eps: float
    :return: geometric median of input points
    :rtype: coordinates of geometric median
    """

    y = np.mean(bmo_points, 0)

    while True:
        d = cdist(bmo_points, y.reshape(1, 2))
        non_zeros = (d != 0)[:, 0]
        d_inv = 1 / d[non_zeros]
        d_inv_sum = np.sum(d_inv)
        w = d_inv / d_inv_sum
        t = np.sum(w * bmo_points[non_zeros], 0)
        num_zeros = len(bmo_points) - np.sum(non_zeros)
        if num_zeros == 0:
            y1 = t
        elif num_zeros == len(bmo_points):
            return y
        else:
            r = (t - y) * d_inv_sum
            r_norm = np.linalg.norm(r)
            r_inv = 0 if r == 0 else num_zeros / r_norm
            y1 = max(0, 1 - r_inv) * t + min(1, r_inv) * y
        if euclidean(y, y1) < eps:
            return y1
        y = y1
