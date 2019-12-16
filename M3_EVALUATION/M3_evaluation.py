import pandas as pd
import numpy as np
import os
import math
import gc
from M3_Adaptive_NN import neural_nets

from tensorflow import set_random_seed

set_random_seed(1000)

def write_res(model_class, final_res, methods, var, data_name):

    trainX, valX, testX, trainY, valY, testY, scalerY = model_class.data_preprocessing()

    for method in methods:
        if method == 'LSTM':
            last_model = model_class.fittingLSTM('models/%s/%s/model_NN_%s.h5' % (data_name, method, var))
        elif method == 'GRU':
            last_model = model_class.fittingGRU('models/%s/%s/model_NN_%s.h5' % (data_name, method, var))
        elif method == 'RNN':
            last_model = model_class.fittingRNN('models/%s/%s/model_NN_%s.h5' % (data_name, method, var))
        else:
            last_model = model_class.fittingMLP('models/%s/%s/model_NN_%s.h5' % (data_name, method, var))

        final_pred_y, final_true_y, final_ssr0 = model.predict(testX, testY, scalerY,
                                                               last_model)

        final_rmse, final_mae, final_mape, final_r_squared = model.evaluation(final_pred_y, final_true_y)

        print(var, final_rmse, final_mae, final_mape, final_r_squared)

        final_res['method'].append(method)
        final_res['Dependent_variable'].append(var)
        final_res['Variables'].append('softmax')
        final_res['lag'].append(4)
        final_res['RMSE'].append(final_rmse[0])
        final_res['MAE'].append(final_mae[0])
        final_res['MAPE'].append(final_mape[0])
        final_res['R2'].append(final_r_squared[0])

    return final_res

batch_size=4


NNs = ['LSTM', 'GRU', 'RNN']

M3C = pd.ExcelFile('M3C.xls')
for sheet in 'M3Year M3Quart M3Month M3Other'.split():
    if sheet=='M3Month':
        M3Month = M3C.parse(sheet).values
        for row in M3Month[1422:1431]:
            final_result = {'method': [], 'Dependent_variable': [],
                            'Variables': [], 'lag': [], 'RMSE': [],
                            'MAE': [], 'MAPE': [], 'R2': []}
            series = row[0]
            print(series)
            N = row[1]
            NF = row[2]
            X = np.expand_dims(np.asarray(row[6:6+N]), axis=1)
            Y = np.expand_dims(np.asarray(row[6:6+N]), axis=1)

            model = neural_nets(X=X, Y=Y, N=NF, epoch_number=500, batch_size=batch_size, learning_rate=0.001,
                                encoder=[12], early_stoppting_patience=300, neurons=[12], lag=4,
                                time_step=4, activation='linear', reg_lambda=0.0001, mult=False)

            final_result = write_res(model_class=model, final_res=final_result, methods=NNs,
                                     var=series, data_name='M3')

            res_final = pd.DataFrame.from_dict(final_result)

            if not os.path.isfile('models/Performance/' + 'M3_results.csv'):
                res_final.to_csv('models/Performance/' + 'M3_results.csv',
                                  encoding='euc-kr', index=False)
            else:
                res_final.to_csv('models/Performance/' + 'M3_results.csv', mode='a',
                                  encoding='euc-kr', index=False)
