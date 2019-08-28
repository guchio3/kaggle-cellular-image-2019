import numpy as np


def MAPE(y_true, y_pred):
    n = len(y_true)
    mape = (100 / n) * np.sum(np.abs((y_true - y_pred) / y_true))
    return mape


def lgb_MSE_approx_obj(y_preds, y_true):
    '''
    '''
    # y_preds = y_preds.get_label().values
    y_true = y_true.get_label().values
    grads = -2 * (y_true - y_preds)
    hess = 2 * y_preds * 0 + 1
#    hess += (hess == 0) * 100
    return grads, hess


def lgb_MAPE_approx_obj(y_preds, y_true):
    '''
    '''
    y_true = y_true.get_label().values
    denom = np.abs(y_true)
    numer = 1. * (y_true < y_preds) - 1. * (y_true > y_preds)
    grads = np.max(denom) * numer / denom
    hess = y_preds * 0.
    hess = hess + 0.05
    return grads, hess
