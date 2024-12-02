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


from libs_tf import ScaleLayer, LSTM_tsetQ, LinearRegression
from hydrodata_tf import DataforIndividual
import cn_loss


## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
caravan_path = 'F:\\hydro_dl_project\\camels_data\\caravan\\global'
working_path = 'F:\\hydro_dl_project'
attrs_path = 'F:\\hydro_dl_project\\camels_data\\s_391attrs.csv'

training_start = '1981-10-01'
training_end = '2000-09-30'
start_date = pd.to_datetime('1981/10/1')
end_date = pd.to_datetime('2000/9/30')
nor_start = '1981-10-01'
nor_end = '2000-09-30'

basin_id = ['1022500',
'1031500',
'1047000',
'1052500',
'1054200',
'1055000',
'1057000',
'1078000',
'1134500',
'1137500',
'1139000',
'1144000',
'1169000',
'1170100',
'1181000',
'1333000',
'1350000',
'1365000',
'1413500',
'1423000',
'1435000',
'1439500',
'1440000',
'1440400',
'1451800',
'1487000',
'1491000',
'1510000',
'1532000',
'1539000',
'1543000',
'1543500',
'1544500',
'1545600',
'1547700',
'1548500',
'1550000',
'1552000',
'1557500',
'1568000',
'1580000',
'1583500',
'1596500',
'1605500',
'1606500',
'1632000',
'1632900',
'1634500',
'1638480',
'1639500',
'1644000',
'1664000',
'1666500',
'1667500',
'2011400',
'2013000',
'2014000',
'2015700',
'2016000',
'2017500',
'2018000',
'2027000',
'2027500',
'2028500',
'2046000',
'2051500',
'2053200',
'2053800',
'2056900',
'2059500',
'2064000',
'2065500',
'2069700',
'2070000',
'2074500',
'2077200',
'2081500',
'2082950',
'2092500',
'2108000',
'2111180',
'2111500',
'2112120',
'2112360',
'2118500',
'2125000',
'2128000',
'2137727',
'2143000',
'2149000',
'2152100',
'2177000',
'2178400',
'2202600',
'2212600',
'2221525',
'2231000',
'2245500',
'2246000',
'2296500',
'2297310',
'2298123',
'2298608',
'2299950',
'2342933',
'2349900',
'2361000',
'2369800',
'2371500',
'2372250',
'2374500',
'2450250',
'2464000',
'2469800',
'2472000',
'2472500',
'2479155',
'2479300',
'2479560',
'2481000',
'2481510',
'3010655',
'3011800',
'3015500',
'3021350',
'3028000',
'3049000',
'3069500',
'3070500',
'3076600',
'3078000',
'3144000',
'3170000',
'3173000',
'3180500',
'3182500',
'3186500',
'3237500',
'3238500',
'3241500',
'3280700',
'3281500',
'3285000',
'3340800',
'3346000',
'3364500',
'3366500',
'3384450',
'3439000',
'3456500',
'3460000',
'3463300',
'3471500',
'3473000',
'3479000',
'3488000',
'3498500',
'3500000',
'3500240',
'3504000',
'3574500',
'3604000',
'4015330',
'4024430',
'4027000',
'4040500',
'4045500',
'4057510',
'4057800',
'4059500',
'4063700',
'4074950',
'4105700',
'4122200',
'4122500',
'4127918',
'4127997',
'4185000',
'4196800',
'4197100',
'4213000',
'4216418',
'4221000',
'4224775',
'4256000',
'4296000',
'5057200',
'5120500',
'5291000',
'5362000',
'5393500',
'5399500',
'5408000',
'5413500',
'5414000',
'5444000',
'5458000',
'5466500',
'5487980',
'5489000',
'5495000',
'5495500',
'5501000',
'5503800',
'5507600',
'5508805',
'5525500',
'5556500',
'5584500',
'5592050',
'5593575',
'5593900',
'5595730',
'6224000',
'6280300',
'6289000',
'6332515',
'6339100',
'6344600',
'6350000',
'6352000',
'6404000',
'6406000',
'6409000',
'6431500',
'6447500',
'6470800',
'6477500',
'6601000',
'6622700',
'6623800',
'6632400',
'6803510',
'6803530',
'6814000',
'6847900',
'6853800',
'6876700',
'6878000',
'6885500',
'6888500',
'6889200',
'6889500',
'6892000',
'6903400',
'6910800',
'6911900',
'6917000',
'6918460',
'6919500',
'6921070',
'6921200',
'7057500',
'7060710',
'7066000',
'7142300',
'7145700',
'7167500',
'7180500',
'7184000',
'7196900',
'7197000',
'7208500',
'7261000',
'7291000',
'7299670',
'7301410',
'7315200',
'7315700',
'7335700',
'7340300',
'7346045',
'7362100',
'7375000',
'8013000',
'8014500',
'8023080',
'8066200',
'8066300',
'8070000',
'8082700',
'8086212',
'8086290',
'8101000',
'8104900',
'8109700',
'8150800',
'8158700',
'8164300',
'8164600',
'8165300',
'8171300',
'8175000',
'8176900',
'8189500',
'8190000',
'8190500',
'8194200',
'8195000',
'8196000',
'8198500',
'8200000',
'8202700',
'8269000',
'8324000',
'8377900',
'8378500',
'8380500',
'9081600',
'9210500',
'9223000',
'9312600',
'9352900',
'9386900',
'9404450',
'9430600',
'9492400',
'9494000',
'9497980',
'9505350',
'9505800',
'9510200',
'9512280',
'9513780',
'10234500',
'11124500',
'11143000',
'11148900',
'11151300',
'11176400',
'11230500',
'11264500',
'11266500',
'11381500',
'11451100',
'11468500',
'11473900',
'11478500',
'11480390',
'11481200',
'11482500',
'11522500',
'11523200',
'11528700',
'11532500',
'12010000',
'12013500',
'12020000',
'12025700',
'12035000',
'12040500',
'12041200',
'12048000',
'12054000',
'12056500',
'12082500',
'12092000',
'12115000',
'12147500',
'12167000',
'12175500',
'12186000',
'12189500',
'12390700',
'12411000',
'12451000',
'12488500',
'13011500',
'13011900',
'13023000',
'13161500',
'13235000',
'13240000',
'13313000',
'13331500',
'14020000',
'14137000',
'14154500',
'14166500',
'14182500',
'14185000',
'14185900',
'14222500',
'14236200',
'14301000',
'14305500',
'14306500',
'14309500',
'14316700',
'14325000',
'14400000',
]

print(len(basin_id))

def normalize(data):
    train_mean = np.mean(data, axis=0, keepdims=True)
    train_std = np.std(data, axis=0, keepdims=True)
    train_scaled = (data - train_mean) / train_std
    return train_scaled, train_mean, train_std

all_list = []
all_list_for_nor = []
for i in range(len(basin_id)):

    if len(basin_id[i]) == 7:
        basin_id[i] = '0' + basin_id[i]
        print(basin_id[i])

    hydrodata = DataforIndividual(working_path, basin_id[i]).load_data()

    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    nor_set = hydrodata[hydrodata.index.isin(pd.date_range(nor_start, nor_end))]
    # print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
    # print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")

    a = basin_id[i]
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
    # print("local_static_x_test:", local_static_x_for_test)
    # print("local_static_x_test_shape:", local_static_x_for_test.shape)

    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(train_set.shape[0], axis=0)

    # print("local_static_x_train_shape:", local_static_x_for_train.shape)
    # print(local_static_x_for_train[0,0])

    result = np.concatenate((train_set, local_static_x_for_train), axis=-1)


    curr_csv_path = f'{caravan_path}\\camels_{str(a)}.csv'
    # Read the CSV file and set the 'date' column as the index
    caravan_basin = pd.read_csv(curr_csv_path, parse_dates=['date'], index_col='date')
    s1 = caravan_basin['volumetric_soil_water_layer_1_mean']
    subs1 = s1[start_date:end_date]
    np_subs1 = subs1.to_numpy()
    np_subs1 = [num * 70 for num in np_subs1]
    np_subs1 = np.array(np_subs1)
    print("np_subs1.shape:", np_subs1.shape)
    np_subs1 = np_subs1.reshape((np_subs1.shape[0], 1))
    print("np_subs1.shape:", np_subs1.shape)

    '''if np.isnan(np_subs1).any():
        print("q_nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", basin_id[i])
    if np.isinf(np_subs1).any():
        print("q_inf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", basin_id[i])'''

    #print("np_subq.shape:",np_subq)
    result = np.concatenate((result, np_subs1), axis=-1)

    # print("result_shape:",result.shape)

    all_list.append(result)
    all_list_for_nor.append(nor_set)

print(len(all_list))
print(len(all_list_for_nor))

result_ = all_list[0]
nor_result_ = all_list_for_nor[0]

for i in range(len(all_list)-1):
    result_ = np.concatenate((result_, all_list[i+1]), axis=0)
print(result_.shape)

#Five meteorological, 27 static attributes, runoff in the last dimension, 34th is soil moisture
sum_result = result_[:,
             [0, 1, 2, 3, 4, 32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
              28, 29, 30, 31, 33, 5]]

for i in range(len(all_list_for_nor)-1):
    nor_result_ = np.concatenate((nor_result_, all_list_for_nor[i+1]), axis=0)
print(nor_result_.shape)

sumnor_result = nor_result_[:,
             [0, 1, 2, 3, 4, 5]]


#generate_train_test(sum_result4+27+1+1, sumnor_result5+1, wrap_length=wrap_length)
def generate_train_test(train_set, all_data, wrap_length):

    all_set_ = pd.DataFrame(all_data)
    all_set_npx = all_set_.values[:, :-1]
    all_set_npx_prcp_mean = np.mean(all_set_npx[:,0:1])
    all_set_npx_tmin_mean = np.mean(all_set_npx[:,1:2])
    all_set_npx_tmax_mean = np.mean(all_set_npx[:,2:3])
    all_set_npx_srad_mean = np.mean(all_set_npx[:,3:4])
    all_set_npx_vp_mean = np.mean(all_set_npx[:,4:5])
    all_set_npx_prcp_std = np.std(all_set_npx[:,0:1])
    all_set_npx_tmin_std = np.std(all_set_npx[:,1:2])
    all_set_npx_tmax_std = np.std(all_set_npx[:,2:3])
    all_set_npx_srad_std = np.std(all_set_npx[:,3:4])
    all_set_npx_vp_std = np.std(all_set_npx[:,4:5])

    train_set_ = pd.DataFrame(train_set)
    train_x_np = train_set_.values[:, :-2]
    train_x_np[:,0:1] = (train_x_np[:,0:1] - all_set_npx_prcp_mean)/all_set_npx_prcp_std
    train_x_np[:,1:2] = (train_x_np[:,1:2] - all_set_npx_tmin_mean)/all_set_npx_tmin_std
    train_x_np[:,2:3] = (train_x_np[:,2:3] - all_set_npx_tmax_mean)/all_set_npx_tmax_std
    train_x_np[:,3:4] = (train_x_np[:,3:4] - all_set_npx_srad_mean)/all_set_npx_srad_std
    train_x_np[:,4:5] = (train_x_np[:,4:5] - all_set_npx_vp_mean)/all_set_npx_vp_std

    print("all_set_npx_prcp_mean_std:", all_set_npx_prcp_mean, all_set_npx_prcp_std)
    print("all_set_npx_tmin_mean_std:", all_set_npx_tmin_mean, all_set_npx_tmin_std)
    print("all_set_npx_tmax_mean_std:", all_set_npx_tmax_mean, all_set_npx_tmax_std)
    print("all_set_npx_srad_mean_std:", all_set_npx_srad_mean, all_set_npx_srad_std)
    print("all_set_npx_vp_mean_std:", all_set_npx_vp_mean, all_set_npx_vp_std)
################################################################################################
    all_set_ = pd.DataFrame(train_set)
    all_set_npy = all_set_.values[:, -2:]
    all_set_npy_s1_mean = np.mean(all_set_npy[:,0:1])
    all_set_npy_q_mean = np.mean(all_set_npy[:,1:2])
    all_set_npy_s1_std = np.std(all_set_npy[:,0:1])
    all_set_npy_q_std = np.std(all_set_npy[:,1:2])

    train_set_ = pd.DataFrame(train_set)
    train_y_np = train_set_.values[:, -2:]
    train_y_np[:,0:1] = (train_y_np[:,0:1] - all_set_npy_s1_mean)/all_set_npy_s1_std
    train_y_np[:,1:2] = (train_y_np[:,1:2] - all_set_npy_q_mean)/all_set_npy_q_std

    print("all_set_npx_s1_mean_std:", all_set_npy_s1_mean, all_set_npy_s1_std)
    print("all_set_npx_q_mean_std:", all_set_npy_q_mean, all_set_npy_q_std)


    wrap_number_train = (train_x_np.shape[0] - wrap_length) // 5 + 1

    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 5:(wrap_length + i * 5), :]
        train_y[i, :, :] = train_y_np[i * 5:(wrap_length + i * 5), :]

    return train_x, train_y


wrap_length = 270  # It can be other values, but recommend this value should not be less than 5 years (1825 days).
train_x, train_y  = generate_train_test(sum_result, sumnor_result, wrap_length=wrap_length)
#train_x, train_y, train_ys1 = generate_train_test(sum_result, nor_result_, wrap_length=wrap_length)
print(f'The shape of train_x, train_y, test_x, and test_y after wrapping by {wrap_length} days are:')
print(f'{train_x.shape}, {train_y.shape}')



def create_model(input_xd_shape, seed):
    xd_input_forprnn = Input(shape=input_xd_shape, batch_size=1167, name='Input_xd1')  # [9,3288,5]
    # xd_input_forconnect = Input(shape=input_xd2_shape, batch_size=9, name='Input_xd2')  #[9,3288,5]
    # xs_input = Input(shape=input_xs_shape, batch_size=15, name='Input_xs')          #[9,27]

    cn, hn = LSTM_tsetQ(input_xd=32, hidden_size=64, seed=seed)(xd_input_forprnn)

    fc_x = Dropout(0.4)(hn)

    fc_out = Dense(units=1)(fc_x)
    fc_out = K.permute_dimensions(fc_out, pattern=(1, 0, 2))

    sn = LinearRegression(output_dim=1, seed=seed)(cn)
    sn = K.permute_dimensions(sn, pattern=(1, 0, 2))

    print("fc_out.shape", fc_out.shape)
    print("sn.shape", sn.shape)
    qsn = Concatenate(axis=-1, name='Concat')([sn,fc_out])

    model = Model(inputs=xd_input_forprnn, outputs=qsn)

    return model

def train_model(model, train_xd, train_y, ep_number, lrate, save_path):
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)

    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)

    tnan = callbacks.TerminateOnNaN()


    def custom_nse_loss(y_true, y_pred, alpha=1.0, rho=0.15, weight_q=0.7, weight_s1=0.3):
        return cn_loss.elastic_nse_loss(y_true, y_pred, alpha, rho, weight_q, weight_s1)

    def custom_nse_metrics(y_true, y_pred, alpha=1.0, rho=0.15, weight_q=0.7, weight_s1=0.3):
        return cn_loss.elastic_nse_metrics(y_true, y_pred, alpha, rho, weight_q, weight_s1)

    # Compile the model
    model.compile(
        loss=custom_nse_loss,
        metrics=[custom_nse_metrics],
        optimizer=tf.keras.optimizers.Adam(learning_rate=lrate)
    )



    history = model.fit(x=train_xd, y=train_y, epochs=ep_number, batch_size=1167,
                        callbacks=[save, es, reduce, tnan])
    return history

Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
save_path_ealstm = f'{working_path}/results/model.h5'

model = create_model(input_xd_shape=(train_x.shape[1], train_x.shape[2]), seed=200)
model.summary()

prnn_ealstm_history = train_model(model=model, train_xd=train_x,
                                  train_y=train_y, ep_number=200, lrate=0.01, save_path=save_path_ealstm)
