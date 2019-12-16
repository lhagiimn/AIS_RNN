import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error


# rows = pd.read_csv('names.csv')
# print(rows)
# rows = rows['Data'].tolist()

test = {}
train = {}
M3C = pd.ExcelFile('M3C.xls')
rows = []
for sheet in 'M3Year M3Quart M3Month M3Other'.split():
    if sheet=='M3Month':
        M3Month = M3C.parse(sheet).values
        for row in M3Month:
            series = row[0]
            N = row[1]
            if N > 81:
                rows.append(series)
                NF = row[2]
                train[series] = row[6:6+N-NF]
                test[series] = row[6+N-NF:6+N]


MAPE_ALL = {}
RMSE_ALL = {}
MAE_ALL = {}
MAPE_ALL['Data']=rows
RMSE_ALL['Data'] = rows
MAE_ALL['Data'] = rows
M3Forecast = pd.ExcelFile('M3Forecast.xls')
for sheet in M3Forecast.sheet_names:
    mape_temp = []
    rmse_temp = []
    mae_temp = []
    method_name=sheet
    m = M3Forecast.parse(sheet, header=None).values
    for row in m:
        series = row[0]
        if series in rows:
            NF = row[1]
            pred = row[2:2+NF]
            true = test[series]
            MAPE = np.mean(abs(pred-true)/(true)*100)
            RMSE = math.sqrt(mean_squared_error(true, pred))
            MAE = mean_absolute_error(true, pred)
            mape_temp.append(MAPE)
            rmse_temp.append(RMSE)
            mae_temp.append(MAE)
    MAPE_ALL[method_name]=mape_temp
    RMSE_ALL[method_name] = rmse_temp
    MAE_ALL[method_name] = mae_temp


res_final = pd.DataFrame.from_dict(MAPE_ALL)

if not os.path.isfile('models/Performance/' + 'M3_results_MAPE_sub.csv'):
    res_final.to_csv('models/Performance/' + 'M3_results_MAPE_sub.csv',
                      encoding='euc-kr', index=False)
else:
    res_final.to_csv('models/Performance/' + 'M3_results_MAPE_sub.csv', mode='a',
                      encoding='euc-kr', index=False)


res_final = pd.DataFrame.from_dict(RMSE_ALL)

if not os.path.isfile('models/Performance/' + 'M3_results_RMSE_sub.csv'):
    res_final.to_csv('models/Performance/' + 'M3_results_RMSE_sub.csv',
                      encoding='euc-kr', index=False)
else:
    res_final.to_csv('models/Performance/' + 'M3_results_RMSE_sub.csv', mode='a',
                      encoding='euc-kr', index=False)


res_final = pd.DataFrame.from_dict(MAE_ALL)

if not os.path.isfile('models/Performance/' + 'M3_results_MAE_sub.csv'):
    res_final.to_csv('models/Performance/' + 'M3_results_MAE_sub.csv',
                      encoding='euc-kr', index=False)
else:
    res_final.to_csv('models/Performance/' + 'M3_results_MAE_sub.csv', mode='a',
                      encoding='euc-kr', index=False)





