"""
A class to interpolate a ring scan with given radius and center (BMO points) from a given img scan.

@author = Martin Tschaikner

@Date = 09.09.2019
"""

# import numerical python, statistics and plot packages
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt


class RingScanFromImg:
    """
    This class computes interpolated ring scan data (image) from a given img scan.
    """

    def __init__(self, b_scan_stack, file_name_bmo, resolution, scan_pos_input, radius, number_circle_points,
                 filter_parameter):

        """
        Declaration of the class variables.

        :param b_scan_stack: volume data of img file [z x y]
        :type b_scan_stack: 3D float array
        :param resolution: resolution of img file data [z x y]
        :type resolution: 1D float array
        :param scan_pos_input: OD/OS position for correct calculation of circle points
        :type scan_pos_input: 1D float array
        :param file_name_bmo: text file of bmo points of corresponding vol file
        :type file_name_bmo: csv text file
        :param radius: radius of ring scan
        :type radius: float - default: 1.75 mm
        :param number_circle_points: number of equidistant circle points on ring scan
        :type number_circle_points: integer
        :param filter_parameter: weighting parameter for ring scan interpolation
        :type filter_parameter: integer
        """

        self.b_scan_stack = b_scan_stack
        self.resolution = resolution
        self.file_name_bmo = file_name_bmo
        self.scan_pos_input = scan_pos_input
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
        # correction for crop within bmo points calculation
        bmo_crop = 15
        # import bmo points, scaling, computing of bmo center as mean of all points and projection on x,y plane
        bmo_data = pd.read_csv(self.file_name_bmo, sep=",", header=None)
        bmo_points = np.zeros([bmo_data.shape[1], bmo_data.shape[0]], dtype=float)
        bmo_points[:, 0] = bmo_data.iloc[0, :] * self.resolution[0]
        bmo_points[:, 1] = (bmo_data.iloc[1, :] + bmo_crop) * self.resolution[1]
        bmo_points[:, 2] = bmo_data.iloc[2, :] * self.resolution[2]
        bmo_center_3d = np.mean(bmo_points, axis=0)
        bmo_center_2d = np.array([bmo_center_3d[0], bmo_center_3d[1]])
        bmo_center_geom_mean_2d = geometric_median(bmo_points[:, 0:2], eps=1e-6)

        # compute noe equidistant circle points around bmo center with given radius
        noe = self.number_circle_points

        # OD clock wise, OS ccw ring scan interpolation for correct orientation
        if self.scan_pos_input == 'OS':
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
        :return: interpolated grey value image
        :rtype: 2d float array
        """

        noe = self.number_circle_points

        # create data arrays for interpolation
        ring_scan_data = np.zeros([np.size(self.b_scan_stack, 0), noe])

        # filter size computed depending on ratio of x to y resolution of volume data
        if self.resolution[0] > self.resolution[1]:
            multiples = np.around(self.resolution[0] / self.resolution[1])
            f_x = int(2 * self.filter_parameter + 1)
            f_y = int(2 * multiples * self.filter_parameter + 1)
            # defines sigma as 1/3 of greater distance
            sigma = 1 / 3 * max(self.resolution[0], multiples * self.resolution[1])
        else:
            multiples = np.around(self.resolution[1] / self.resolution[0])
            f_x = int(2 * multiples * self.filter_parameter + 1)
            f_y = int(2 * self.filter_parameter + 1)
            # defines sigma as 1/3 of greater distance
            sigma = 1 / 3 * max(multiples * self.resolution[0], self.resolution[1])

        # sigma^2
        if int(self.filter_parameter) != 0:
            sigma2 = np.square(self.filter_parameter * sigma)
        else:
            # defines sigma as 1/3 of smaller distance
            sigma2 = np.square(1 / 3 * min(self.resolution[0], self.resolution[1]))

        # loop over all circle points to gain interpolated values
        for i in range(noe):
            # reshape and calculating indices of nearest data grid point to ith circle point
            loc_var = np.reshape(circle_points_coordinates[:, i], [2, 1])
            index_x_0 = np.around(loc_var[0] / self.resolution[0]).astype(int)
            index_y_0 = np.around(loc_var[1] / self.resolution[1]).astype(int)

            # compute range of indices for filter mask
            index_x_min = int(index_x_0 - (f_x - 1) / 2)
            index_x_max = int(index_x_0 + (f_x - 1) / 2)
            index_y_min = int(index_y_0 - (f_y - 1) / 2)
            index_y_max = int(index_y_0 + (f_y - 1) / 2)

            # fill filter mask with corresponding indices
            index_xx, index_yy = np.meshgrid(np.linspace(index_x_min, index_x_max, f_x),
                                             np.linspace(index_y_min, index_y_max, f_y))

            # compute matrix of indices differences in x, y direction
            diff_x = index_xx - loc_var[0] / self.resolution[0]
            diff_y = index_yy - loc_var[1] / self.resolution[1]

            # compute weights and interpolated grey values
            w = np.exp(-(np.square(diff_x * self.resolution[0]) +
                         np.square(diff_y * self.resolution[1])) / (2 * sigma2))

            # get gray values within filter mask from volume data
            gv = self.b_scan_stack[:, index_y_min:index_y_max + 1, index_x_min:index_x_max + 1]

            # apply weights at grey values and compute interpolated value
            gv_w = np.sum(np.sum(w * gv, axis=1), axis=1) / np.sum(w)

            # set grey values greater 260 to 0
            gv_w[gv_w >= 0.260e3] = 0

            # fill ring scan data array
            ring_scan_data[:, i] = gv_w

        return ring_scan_data


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
