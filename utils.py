import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def nn(x_test, y_test, y_pred, x_train, y_train):
    """
    Returns the nearest neighbor metric on time series predictions. The metric
    is a percentage of time when the prediction (by some model) is closer to
    ground truth than a nearest neighbor from the training set.
    """
    assert x_test.shape[0] == y_test.shape[0]
    assert x_test.shape[0] == y_pred.shape[0]
    assert x_train.shape[0] == y_train.shape[0]
    
    model_closer = 0

    for i in tqdm(range(x_test.shape[0])):
        x = x_test[i]
        y_true = y_test[i]
        nn_idx = np.argmin(np.linalg.norm(x_train - x, axis=1))
        nn_distance, _ = fastdtw(y_train[nn_idx], y_true, dist=euclidean)
        pred_distance, _ = fastdtw(y_pred[i], y_true, dist=euclidean)
        if pred_distance < nn_distance:
            model_closer += 1
    return model_closer / x_test.shape[0]
