import keras.backend as K
import numpy as np
from scipy import stats
import logging
import tensorflow as tf
import tensorflow.keras.backend as K

def elastic_nse_loss(y_true, y_pred, alpha=1.0, rho=0.15, weight_q=0.5, weight_s1=0.5):

    # Extract flow and hydrological conceptual variables
    q_true = y_true[:, :, -1]
    q_pred = y_pred[:, :, -1]
    s1_true = y_true[:, :, -2]
    s1_pred = y_pred[:, :, -2]

    # Calculating NSE Loss
    q_numerator = K.sum(K.square(q_pred - q_true), axis=1)
    q_denominator = K.sum(K.square(q_true - K.mean(q_true, axis=1, keepdims=True)), axis=1)
    q_loss = q_numerator / (q_denominator + 0.5)

    s1_numerator = K.sum(K.square(s1_pred - s1_true), axis=1)
    s1_denominator = K.sum(K.square(s1_true - K.mean(s1_true, axis=1, keepdims=True)), axis=1)
    s1_loss = s1_numerator / (s1_denominator + 0.5)

    # Weighted combined target variable loss
    combined_loss = weight_q * q_loss + weight_s1 * s1_loss

    # Adding Elastic Net Regularization
    weights = K.flatten(y_pred)  #
    l1_penalty = K.sum(K.abs(weights))
    l2_penalty = K.sum(K.square(weights))
    elastic_net_penalty = alpha * (rho * l1_penalty + (1 - rho) * l2_penalty)

    # the total loss
    total_loss = K.mean(combined_loss) + elastic_net_penalty
    return total_loss

def elastic_nse_metrics(y_true, y_pred, alpha=1.0, rho=0.15, weight_q=0.5, weight_s1=0.5):
    # Extract flow and hydrological conceptual variables
    q_true = y_true[:, :, -1]
    q_pred = y_pred[:, :, -1]
    s1_true = y_true[:, :, -2]
    s1_pred = y_pred[:, :, -2]
    # Calculating NSE Loss
    q_numerator = K.sum(K.square(q_pred - q_true), axis=1)
    q_denominator = K.sum(K.square(q_true - K.mean(q_true, axis=1, keepdims=True)), axis=1)
    q_nse = 1 - q_numerator / (q_denominator + 0.5)

    s1_numerator = K.sum(K.square(s1_pred - s1_true), axis=1)
    s1_denominator = K.sum(K.square(s1_true - K.mean(s1_true, axis=1, keepdims=True)), axis=1)
    s1_nse = 1 - s1_numerator / (s1_denominator + 0.5)

    # Weighted composite target variable NSE
    combined_nse = weight_q * q_nse + weight_s1 * s1_nse

    # Added elastic net regularization (only monitors the regularization value, not involved in the final metric calculation)
    weights = K.flatten(y_pred)
    l1_penalty = K.sum(K.abs(weights))
    l2_penalty = K.sum(K.square(weights))
    elastic_net_penalty = alpha * (rho * l1_penalty + (1 - rho) * l2_penalty)

    # Total indicators (remove the regularization term and only show the prediction effect of training data)
    final_metric = K.mean(combined_nse)
    return final_metric


def nse_loss(y_true, y_pred):
    #y_pred1 = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]
    y_pred1 = y_pred
    s1_true = y_true[:, :, -2]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    s1_pred = y_pred1[:, :,-2]  # Omit values in the spinup period (the first 365 days)
    q_true = y_true[:, :, -1]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    q_pred = y_pred1[:, :, -1]  # Omit values in the spinup period (the first 365 days)"""

    #NSE
    #eps = 1e-3
    q_numerator = K.sum(K.square(q_pred - q_true), axis=1)  #
    q_denominator = K.sum(K.square(q_true - K.mean(q_true, axis=1, keepdims=True)), axis=1)  #
    q_loss = q_numerator / (q_denominator+0.5)
    #NSE
    s1_numerator = K.sum(K.square(s1_pred - s1_true), axis=1)  #
    s1_denominator = K.sum(K.square(s1_true - K.mean(s1_true, axis=1, keepdims=True)), axis=1)  #
    s1_loss = s1_numerator / s1_denominator

    sum_loss = (q_loss + s1_loss) / 2

    return sum_loss

def nse_metrics(y_true, y_pred):
    #y_pred1 = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]
    y_pred1 = y_pred
    q_true = y_true[:, :, -1]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    q_pred = y_pred1[:, :, -1]  # Omit values in the spinup period (the first 365 days)
    s1_true = y_true[:, :, -2]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    s1_pred = y_pred1[:, :, -2]  # Omit values in the spinup period (the first 365 days)"""

    #NSE
    #eps = 1e-3
    q_numerator = K.sum(K.square(q_pred - q_true), axis=1)  #
    q_denominator = K.sum(K.square(q_true - K.mean(q_true, axis=1, keepdims=True)), axis=1)  #
    q_loss = q_numerator / (q_denominator+0.5)

    #NSE
    s1_numerator = K.sum(K.square(s1_pred - s1_true), axis=1)  #
    s1_denominator = K.sum(K.square(s1_true - K.mean(s1_true, axis=1, keepdims=True)), axis=1)  #
    s1_loss = s1_numerator / (s1_denominator)

    sum_loss = (q_loss + s1_loss) / 2

    return 1 - sum_loss


def r_squared_loss(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r_squared = 1 - SS_res / (SS_tot + K.epsilon())

    return r_squared


def r_loss(y_true, y_pred):
    y_true_mean = K.mean(y_true)
    y_pred_mean = K.mean(y_pred)

    numerator = K.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = K.sqrt(K.sum(K.square(y_true - y_true_mean)) * K.sum(K.square(y_pred - y_pred_mean)))

    r = numerator / (denominator + K.epsilon())

    return -r

def negative_nse_loss(y_true, y_pred):

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    y_pred = y_pred[:, :,:]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    nse = 1 - (numerator / (denominator + K.epsilon()))  # 使用相关系数的负值作为损失函数

    return K.mean(nse)



def mse_loss(y_true, y_pred):

    #y_pred1 = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]

    y_pred1 = y_pred
    q_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    q_pred = y_pred1[:, :,:]  # Omit values in the spinup period (the first 365 days)

    #MSE
    mse_q = K.mean(K.square(q_pred - q_true), axis=1)
    #mse_s0 = K.mean(K.square(s0_true - s0_pred), axis=1)
    #mse_s1 = K.mean(K.square(s1_true - s1_pred), axis=1)

    #sum_loss = mse_q + mse_s0 + mse_s1
    #sum_loss = (q_loss + s0_loss + s1_loss)/3
    sum_loss = mse_q


    return sum_loss


def mse_metrics(y_true, y_pred):

    #y_pred1 = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]
    y_pred1 = y_pred
    q_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    q_pred = y_pred1[:, :, :]  # Omit values in the spinup period (the first 365 days)

    #MSE
    mse_q = K.mean(K.square(q_pred - q_true), axis=1)

    sum_loss = mse_q

    return 1 - sum_loss