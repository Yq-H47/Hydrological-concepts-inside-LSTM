import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model

from keras.layers import Layer, Dense, Lambda, Activation, Input, Concatenate, Dropout
from keras import optimizers, callbacks, optimizers, callbacks, losses, metrics
#from matplotlib import pyplot as plt
from datetime import datetime, timedelta

import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf


from libs_tf import ScaleLayer, LSTM, physics_LSTM
from hydrodata_tf import DataforIndividual
import cn_loss


## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

working_path = '\your_path'
attrs_path = '\s_391attrs.csv'

testing_start = '2005-10-1'
testing_end = '2010-09-30'


basin_id = [
'1022500',
'1031500',
'1047000',
'1057000',
'1055000',
'1052500',
'1137500',
'1134500',
'1078000',
'1139000',
'4296000',
'1144000',
'1170100',
'1169000',
'1181000',
'1333000',
'1350000',
'1365000',
'1435000',
'1413500',
'1440000',
'1423000',
'1439500',
'4256000',
'1440400',
'1487000',
'1451800',
'1491000',
'1510000',
'1539000',
'1580000',
'1552000',
'1532000',
'1583500',
'1550000',
'1639500',
'2053200',
'1568000',
'1548500',
'2092500',
'1545600',
'1544500',
'2046000',
'1547700',
'1644000',
'4224775',
'4221000',
'2108000',
'2082950',
'3010655',
'1664000',
'1557500',
'2051500',
'1667500',
'4216418',
'1543000',
'1543500',
'1634500',
'3011800',
'3028000',
'2081500',
'1632900',
'2065500',
'2028500',
'1632000',
'2064000',
'2027000',
'1596500',
'2027500',
'3078000',
'2077200',
'1606500',
'1605500',
'3015500',
'2015700',
'2016000',
'2074500',
'3070500',
'3069500',
'2059500',
'3021350',
'3049000',
'2011400',
'3180500',
'2128000',
'3182500',
'2056900',
'2070000',
'2013000',
'2018000',
'2014000',
'2069700',
'2053800',
'3186500',
'2017500',
'2125000',
'3170000',
'4213000',
'2112360',
'2118500',
'3173000',
'2111500',
'3471500',
'2111180',
'3488000',
'2143000',
'2296500',
'2202600',
'2298123',
'3473000',
'2152100',
'3479000',
'2245500',
'2246000',
'2297310',
'2298608',
'2299950',
'2137727',
'3144000',
'3463300',
'2231000',
'3456500',
'3439000',
'4197100',
'3460000',
'2177000',
'3280700',
'3500000',
'4196800',
'3500240',
'3237500',
'2178400',
'3504000',
'2221525',
'3281500',
'2212600',
'3241500',
'3498500',
'3238500',
'2349900',
'4185000',
'3285000',
'4127997',
'4127918',
'4105700',
'4045500',
'2361000',
'3366500',
'3364500',
'4122500',
'4122200',
'2371500',
'3574500',
'2372250',
'2369800',
'4057510',
'3340800',
'2374500',
'2450250',
'3604000',
'4059500',
'2464000',
'5525500',
'2469800',
'4057800',
'3346000',
'2479560',
'4040500',
'3384450',
'4063700',
'5592050',
'4074950',
'2479155',
'5595730',
'2481000',
'5556500',
'5593900',
'5593575',
'2481510',
'2472000',
'2472500',
'5444000',
'5393500',
'5399500',
'7375000',
'5362000',
'5466500',
'5408000',
'4027000',
'7291000',
'5413500',
'5584500',
'5495500',
'5508805',
'7066000',
'5507600',
'4015330',
'5501000',
'7057500',
'5495000',
'5503800',
'7261000',
'4024430',
'5458000',
'8013000',
'8014500',
'7362100',
'5489000',
'6921200',
'6921070',
'6903400',
'5487980',
'6918460',
'8023080',
'6919500',
'7340300',
'7196900',
'7346045',
'7335700',
'7197000',
'8066300',
'8066200',
'7184000',
'6917000',
'6892000',
'8070000',
'6889500',
'6889200',
'6911900',
'6910800',
'6814000',
'6888500',
'6885500',
'7167500',
'6601000',
'6803530',
'6803510',
'7180500',
'5291000',
'8164300',
'8164600',
'8109700',
'6878000',
'8176900',
'7145700',
'8189500',
'8175000',
'7315700',
'8104900',
'6876700',
'8158700',
'7315200',
'6470800',
'8101000',
'8171300',
'5057200',
'6853800',
'6477500',
'8086290',
'8194200',
'8150800',
'8200000',
'7142300',
'8086212',
'8202700',
'8198500',
'8165300',
'8082700',
'8195000',
'8196000',
'7299670'
]
print(len(basin_id))

def normalize_minmax(data):
    data_min = np.min(data)
    print("data_min:",data_min)
    data_max = np.max(data)
    data_scaled = (data - data_min) / (data_max - data_min)
    return data_scaled

def normalize(data):
    train_mean = np.mean(data, axis=0, keepdims=True)
    train_std = np.std(data, axis=0, keepdims=True)
    train_scaled = (data - train_mean) / train_std
    return train_scaled


def create_model(input_xd_shape, seed):
    xd_input_forprnn = Input(shape=input_xd_shape, batch_size=1, name='Input_xd1')  # [9,3288,5]
    # xs_input = Input(shape=input_xs_shape, batch_size=15, name='Input_xs')          #[9,27]

    hn = LSTM(input_xd=32, hidden_size=64, seed=seed)(xd_input_forprnn)

    fc_x = Dropout(0.4)(hn)

    fc_out = Dense(units=1)(fc_x)
    #fc_out = K.permute_dimensions(fc_out, pattern=(1, 0, 2))
    print("fc_out.shape", fc_out.shape)

    model = Model(inputs=xd_input_forprnn, outputs=fc_out)

    return model


def stest_model(model, test_xd, save_path):

    model.load_weights(save_path, by_name=True)
    pred_y = model.predict(x=test_xd, batch_size=1)
    return pred_y


def nse_metrics(y_true, y_pred):
    y_true = K.constant(y_true)
    #y_pred = y_pred  # Omit values in the spinup period (the first 365 days)

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    y_pred = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)
    print("y_true_shape:", y_true.shape)
    print("y_pred_shape:", y_pred.shape)

    numerator = K.sum(K.square(y_true - y_pred), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE = numerator / (denominator+0.1)

    return 1.0 - rNSE

def generate_train_test(test_set):

    test_set_ = pd.DataFrame(test_set)
    test_x_np = test_set_.values[:, :-1]

    test_x_np[:, 0:1] = (test_x_np[:, 0:1] - 3.2754975723725033) / 6.8470060441102785
    test_x_np[:, 1:2] = (test_x_np[:, 1:2] - 4.563945789070199) / 9.321139980043302
    test_x_np[:, 2:3] = (test_x_np[:, 2:3] - 16.54220651261271) / 10.0588330269492
    test_x_np[:, 3:4] = (test_x_np[:, 3:4] - 332.4399730588107) / 115.48482574345249
    test_x_np[:, 4:5] = (test_x_np[:, 4:5] - 1000.0919263680502) / 583.4867477813818

    test_y_np = test_set_.values[:, -1:]
    test_y_np_nor = (test_y_np[:, -1:] - 1.3201125400129736) / 2.6578904563752346

    #tes_y_np1 = normalize(test_y_np)
    # test_x_np = test_set.values[:, :-1]
    # test_y_np = test_set.values[:, -1:]

    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np_nor, axis=0)

    return test_x, test_y


basin_list = []
test1_list = []
test2_list = []
batch_list = []
testx_list = []
testy_list = []
all_list = []
for i in range(len(basin_id)):

    a = basin_id[i]
    if len(basin_id[i]) == 7:
        basin_id[i] = '0' + basin_id[i]
        print(basin_id[i])

    hydrodata = DataforIndividual(working_path, basin_id[i]).load_data()
    # train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]
    # print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
    # print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")

    if a.startswith('0'):
        single_basin_id = a[1:]

    else:
        single_basin_id = a

    # print(single_basin_id)

    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('basin_id')
    rows_bool = (static_x.index == int(single_basin_id))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)
    # print("static_x_np_shape:", static_x_np.shape)

    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_test = local_static_x_for_test.repeat(test_set.shape[0], axis=0)
    # print("local_static_x_test:", local_static_x_for_test)
    print("local_static_x_test_shape:", local_static_x_for_test.shape)

    # local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    # local_static_x_for_train = local_static_x_for_train.repeat(train_set.shape[0], axis=0)

    # print("local_static_x_train_shape:", local_static_x_for_train.shape)
    # print(local_static_x_for_train[0,0])

    result = np.concatenate((test_set, local_static_x_for_test), axis=-1)

    # all_list.append(result)

    print("result_shape:", result.shape)

    sum_result = result[:,
                 [0, 1, 2, 3, 4, 32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                  28, 29, 30, 31, 5]]

    #nan_rows = np.isnan(sum_result).any(axis=1)
    #sum_result = sum_result[~nan_rows]

    print("sum_result_shape", sum_result.shape)
    test_x, test_y = generate_train_test(sum_result)

    print(f'{test_x.shape}, {test_y.shape}')
    basin_list.append(basin_id[i])
    batch_list.append(test_x.shape[0])
    test1_list.append(test_x.shape[1])
    test2_list.append(test_x.shape[2])

    testx_list.append(test_x)
    testy_list.append(test_y)

print(len(batch_list))
print(len(test1_list))
print(len(test2_list))
print(len(testx_list))
print(len(testy_list))

nse_results = pd.DataFrame(columns=['Basin_ID', 'NSE_TEST'])


Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
load_path_ealstm = f'result_path/model.h5'
model = create_model((1826, 32), seed=200)
model.load_weights(load_path_ealstm, by_name=True)
lstm_layer = Model(inputs=model.get_layer('lstm').input, outputs=model.get_layer('lstm').output)
lstm_layer.summary()


for j in range(len(testx_list)):
    lstm_layer_output = lstm_layer.predict(testx_list[0])
    print("lstm_layer_output.shape", lstm_layer_output.shape)

    '''for i in range(3):
        for j in range(7305):
            hn = K.eval(lstm_layer_output[:, j, i])
            print(f"hn_{i}:", float(hn))'''

    lstm_layer_output= lstm_layer_output.reshape(1826,64)
    df= pd.DataFrame(lstm_layer_output,columns=['cn_1','cn_2','cn_3','cn_4','cn_5','cn_6','cn_7','cn_8','cn_9','cn_10','cn_11','cn_12','cn_13','cn_14','cn_15','cn_16',
                                                'cn_17','cn_18','cn_19','cn_20','cn_21','cn_22','cn_23','cn_24','cn_25','cn_26','cn_27','cn_28','cn_29','cn_30','cn_31','cn_32',
                                                'cn_33','cn_34','cn_35','cn_36','cn_37','cn_38','cn_39','cn_40','cn_41','cn_42','cn_43','cn_44','cn_45','cn_46','cn_47','cn_48',
                                                'cn_49','cn_50','cn_51','cn_52','cn_53','cn_54','cn_55','cn_56','cn_57','cn_58','cn_59','cn_60','cn_61','cn_62','cn_63','cn_64',
])

    df.to_csv(f'cn_path/Cn_{basin_id[j]}.csv', header=True, index=False)

'''
for j in range(len(testx_list)):

    flow = stest_model(model=model, test_xd=testx_list[j], save_path=load_path_ealstm)
    nse_test = nse_metrics(testy_list[j], flow)
    nse_results = nse_results.append({'Basin_ID': basin_id[j],
                                      'NSE_TEST': np.squeeze(K.eval(nse_test))},
                                     ignore_index=True)

    print(f"{basin_id[j]}_nse:", K.eval(nse_test))'''


