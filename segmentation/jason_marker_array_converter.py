"""
Functions to convert a 2D segmentation array of ILM, RNFL and RPE into a json marker file as used in
OCT marker tool and vice versa.

"""
import numpy as np
import json


def array_to_json(all_segmentation, marker_file, new_file_name):
    """

    Segmentation to marker fct.

    :param all_segmentation: segmentation data
    :type all_segmentation: 2d float array , default (3, 768)
    :param marker_file: path to json file
    :type marker_file: txt file dictionary
    :param new_file_name: new file name for oct marker file
    :type new_file_name: string
    :return: json file with new segmentation data
    :rtype: txt file dictionary
    """

    with open(marker_file, "r") as read_file:
        marker_data = json.load(read_file)

    # rename input data
    ilm = all_segmentation[0]
    rnfl = all_segmentation[1]
    if np.size(all_segmentation, 0) > 2:
        rpe = all_segmentation[2]
    else:
        rpe = rnfl - rnfl

    # initialize string for json input
    ilm_str, rnfl_str, rpe_str = '', '', ''

    # add up strings to import into json dict
    for k in range(np.size(all_segmentation, 1)):
        ilm_str = ilm_str + "%i" % ilm[k] + ' '
        rnfl_str = rnfl_str + "%i" % rnfl[k] + ' '
        rpe_str = rpe_str + "%i" % rpe[k] + ' '

    # save info in dictionary
    marker_data['OctMarker']['Markers']['Patient']['Study']['Series'] \
        ['LayerSegmentation']['BScan']['Lines']['ILM'] = ilm_str
    marker_data['OctMarker']['Markers']['Patient']['Study']['Series'] \
        ['LayerSegmentation']['BScan']['Lines']['RNFL'] = rnfl_str
    marker_data['OctMarker']['Markers']['Patient']['Study']['Series'] \
        ['LayerSegmentation']['BScan']['Lines']['RPE'] = rpe_str

    # save to json
    with open(new_file_name, 'w') as outfile:
        json.dump(marker_data, outfile)

    return


def json_to_array(marker_file):
    """

    Marker to segmentation fct.

    :param marker_file: path to json file
    :type marker_file: txt file dictionary
    :return: segmentation data
    :rtype: 2d float array , default (3, 768)
    """

    with open(marker_file, "r") as read_file:
        marker_data = json.load(read_file)

    # get string info from dictionary
    ilm_str = marker_data['OctMarker']['Markers']['Patient']['Study']['Series'] \
        ['LayerSegmentation']['BScan']['Lines']['ILM']
    rnfl_str = marker_data['OctMarker']['Markers']['Patient']['Study']['Series'] \
        ['LayerSegmentation']['BScan']['Lines']['RNFL']
    rpe_str = marker_data['OctMarker']['Markers']['Patient']['Study']['Series'] \
        ['LayerSegmentation']['BScan']['Lines']['RPE']

    # strings to float tuples
    ilm = np.array(ilm_str.split(' '))[:-1].astype(np.float)
    rnfl = np.array(rnfl_str.split(' '))[:-1].astype(np.float)
    rpe = np.array(rpe_str.split(' '))[:-1].astype(np.float)

    # placeholder segmentation and inserting data
    all_segmentation = np.empty((3, np.size(ilm, 0)))
    all_segmentation[0] = ilm
    all_segmentation[1] = rnfl
    all_segmentation[2] = rpe

    return all_segmentation
