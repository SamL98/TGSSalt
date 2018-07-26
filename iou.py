import numpy as np
from skimage.morphology import label

"""
Return the evaluation metric for a single ground truth, predicted mask pair
:param y: the ground truth image
:param pred: the predicted mask
"""
def iou_metric(y, pred):
    if np.count_nonzero(y) == np.count_nonzero(pred) and np.count_nonzero(y) == 0:
        return 1.0

    # Get the connected components, thresholding each to 0.5
    labels = label(y > 0.5)
    pred = label(pred > 0.5)

    # Get the number of components in each set of images
    true_objs = len(np.unique(labels))
    pred_objs = len(np.unique(pred))

    # This holds the area of the intersection for each component
    # in the true and predicted objects
    intersection = np.histogram2d(labels.flatten(), pred.flatten(), bins=(true_objs, pred_objs))[0]

    # Histogram gets the area because labels (the components) will be grouped
    # into objs bins or components and their counts will be the number of
    # labels with a given label in the components image
    #
    # In other words, this holds the area for each component
    true_area = np.histogram(labels, bins=true_objs)[0]
    pred_area = np.histogram(pred, bins=pred_objs)[0]

    true_area = np.expand_dims(true_area, -1)
    pred_area = np.expand_dims(pred_area, 0)

    # |T U P| = |T| + |P| - |T N P|
    # Holds the area of the union for each component in the true and pred objs
    union = true_area + pred_area - intersection

    # Get rid of the background from both the intersection and the union
    intersection = intersection[1:,1:]
    union = union[1:,1:]

    # Make sure we aren't dividing by 0
    union[union == 0] = 1e-9

    # Compute to IoU
    iou = intersection / union

    def precision(t, iou):
        # Matches is a true_objs x pred_objs array
        # that holds boolean flags as to whether the iou for
        # that component is greater than the given threshold
        matches = iou > t

        # True positives, where pred_objs is True, or its IoU
        # was greater than t. In other words, a nucleus was found
        tp = np.sum(matches, axis=1) == 1

        # False positives, where there was an object in pred
        # but it is not found in y. In other words, a non-nucleus was found
        fp = np.sum(matches, axis=0) == 0

        # False negatives, where pred_objs is False, or a nucleus was not found
        fn = np.sum(matches, axis=1) == 0

        return np.sum(tp), np.sum(fp), np.sum(fn)

    #'''
    precs = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision(t, iou)
        if (tp+fp+fn) > 0:
            precs.append(tp / float(tp+fp+fn))
        else:
            precs.append(0)

    return np.mean(precs)
    #'''
    '''
    tp, fp, fn = precision(0.9, iou)
    if tp+fp+fn == 0:
        return 0
    return tp/float(tp+fp+fn)
    #'''

"""
Return the mean evaluation metric over a batch of images
:param y_batch: the batch of ground truth images
:param pred_batch: the batch of predicted masks
"""
def batch_iou(y_batch, pred_batch):
    batch_size = y_batch.shape[0]
    metric = []
    for batch in range(batch_size):
        y_true = y_batch[batch,:,:,0]
        metric.append(iou_metric(y_true, pred_batch[batch,:,:,0]))
    return np.array(np.mean(metric), dtype=np.float32)
