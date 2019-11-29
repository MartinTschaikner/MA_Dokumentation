import numpy as np


def dice_sc(y_true, y_prediction):
    s = 1.
    y_true_f = y_true.ravel()
    y_prediction_f = y_prediction.ravel()
    intersection = np.sum(y_true_f * y_prediction_f)
    dice_score = (2. * intersection + s) / (np.sum(y_true_f) + np.sum(y_prediction_f) + s)

    return dice_score


def accuracy(y_true, y_prediction):
    y_true_f = y_true.ravel()
    y_prediction_f = y_prediction.ravel()
    compare = y_true_f == y_prediction_f
    accuracy = np.sum(compare)/np.size(y_true_f)

    return accuracy


def precision(y_true, y_prediction):
    """
    Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.

    :param y_true:
    :param y_pred:
    :return:
    """
    epsilon = 0.0001
    y_true_f = y_true.ravel()
    y_prediction_f = y_prediction.ravel()
    true_positives = np.sum(y_true_f * y_prediction_f)
    predicted_positives = np.sum(y_prediction_f)
    precision = true_positives / (predicted_positives + epsilon)

    return precision


def sensitivity(y_true, y_prediction):
    epsilon = 0.0001
    y_true_f = y_true.ravel()
    y_prediction_f = y_prediction.ravel()
    true_positives = np.sum(y_true_f * y_prediction_f)
    ground_truth_positives = np.sum(y_true_f)
    sensitivity = true_positives / (ground_truth_positives + epsilon)

    return sensitivity


def specificity(y_true, y_prediction):
    epsilon = 0.0001
    y_true_f = y_true.ravel().astype('bool')
    y_prediction_f = y_prediction.ravel().astype('bool')
    true_negatives = np.sum(np.invert(y_true_f) * np.invert(y_prediction_f))
    ground_truth_negatives = np.sum(np.invert(y_true_f))
    specificity = true_negatives / (ground_truth_negatives + epsilon)

    return specificity


def balanced_accuracy(y_true, y_prediction):
    spec = specificity(y_true, y_prediction)
    sens = sensitivity(y_true, y_prediction)
    balanced_accuracy = (spec + sens)/2

    return balanced_accuracy

def jaccard(y_true, y_prediction):
    y_true_f = y_true.ravel()
    y_pred_f = y_prediction.ravel()
    intersection = np.sum(y_true_f * y_pred_f)
    sum_ = np.sum(y_true_f + y_pred_f)
    jac_score = intersection / (sum_ - intersection)

    return jac_score


def tversky7(y_true, y_prediction):
    y_true_f = y_true.ravel()
    y_prediction_f = y_prediction.ravel()
    true_pos = np.sum(y_true_f * y_prediction_f)
    false_neg = np.sum(y_true_f * (1-y_prediction_f))
    false_pos = np.sum((1-y_true_f)*y_prediction_f)
    alpha = 0.7
    tversky7_score = true_pos/(true_pos + alpha*false_neg + (1-alpha)*false_pos)

    return tversky7_score


def un_signed_error(segmentation, ground_truth):
    # placeholder pixel difference
    diff = np.empty(segmentation.shape)
    for k in range(np.size(segmentation, 0)):
        diff[k, :] = segmentation[k, :] - ground_truth[k, :]

    signed_error = np.average(diff, axis=1)
    unsigned_error = np.average(np.absolute(diff), axis=1)

    return signed_error, unsigned_error
