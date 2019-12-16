import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN, \
    TimeDistributed, Input, Multiply, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import adam
from keras.models import load_model
from scipy.stats import f
from keras import backend as K
from keras.models import Model


from tensorflow import set_random_seed


def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))


import warnings

warnings.filterwarnings("ignore")

set_random_seed(100)


def data_splitting(X, Y, N, validation=True):
    #### data partitioning
    if validation == True:
        train_len = int((len(Y)-N)*0.9)
        val_len = len(Y)-N
        trainX, valX, testX = X[:train_len], X[train_len:val_len], X[len(Y)-N:]
        trainY, valY, testY = Y[:train_len], Y[train_len:val_len], Y[len(Y)-N:]

        return trainX, valX, testX, trainY, valY, testY
    else:
        train_len = int(np.round(len(Y) * 0.8, 0))
        trainX, testX = X[:train_len], X[train_len:]
        trainY, testY = Y[:train_len], Y[train_len:]

        return trainX, testX, trainY, testY


#### prepare time series for LSTM
def prepare_timeseries(X, Y, lag, mult=True):
    dataX, dataY = [], []
    length = len(Y) - lag
    for i in range(0, length):
        dataX.append(X[i:(i + lag)])
        if mult == True:
            dataY.append(Y[(i + 1):(i + lag + 1)])
        else:
            dataY.append(Y[(i + lag)])
    return np.array(dataX), np.array(dataY)


def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]

    return (flattened_X)



class neural_nets:

    def __init__(self, X, Y, N, epoch_number, batch_size, learning_rate, encoder,
                 early_stoppting_patience, neurons, lag, time_step, activation,
                 reg_lambda=0.0001, mult=False):

        self.X = X
        self.Y = Y
        self.epoch = epoch_number
        self.batch = batch_size
        self.lr = learning_rate
        self.early_stopping = early_stoppting_patience
        self.neurons = neurons
        self.encoder = encoder
        self.lag = lag
        self.time_step = time_step
        self.array_size = X.shape[1]
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.mult = mult
        self.N=N

    def data_preprocessing(self):

        if len(self.X.shape) < 2:
            self.X = np.expand_dims(self.X, axis=1)

        scaler = MinMaxScaler()
        scaler = scaler.fit(self.X)
        scaled_X = scaler.transform(self.X)

        if len(self.Y.shape) < 2:
            self.Y = np.expand_dims(self.Y, axis=1)

        scalerY = MinMaxScaler()
        scalerY = scalerY.fit(self.Y)
        scaled_Y = scalerY.transform(self.Y)

        if self.lag == 0:
            dataX = scaled_X
            dataY = scaled_Y
            self.lag = 1
        else:
            dataX, dataY = prepare_timeseries(scaled_X, scaled_Y, self.lag, mult=self.mult)

        trainX, valX, testX, trainY, valY, testY = data_splitting(dataX, dataY, self.N, validation=True)

        trainX, valX, testX = np.reshape(trainX, (trainX.shape[0], self.time_step, self.array_size)), \
                              np.reshape(valX, (valX.shape[0], self.time_step, self.array_size)), \
                              np.reshape(testX, (testX.shape[0], self.time_step, self.array_size))

        return trainX, valX, testX, trainY, valY, testY, scalerY

    def fittingLSTM(self, model_path):

        trainX, valX, testX, trainY, valY, testY, scalerY = self.data_preprocessing()

        if os.path.isfile(model_path):
            final_model = load_model(model_path)
        else:
            input_shape = (self.time_step, self.array_size)
            input = Input(input_shape)
            Input_selection = Sequential()

            for layer_size in self.encoder:
                Input_selection.add(LSTM(layer_size, activation='linear', return_sequences=True,
                                         kernel_regularizer=regularizers.l1(self.reg_lambda),
                                         recurrent_regularizer=regularizers.l1(self.reg_lambda),
                                         input_shape=input_shape))

            Input_selection.add(TimeDistributed(Dense(self.array_size, kernel_regularizer=regularizers.l1(self.reg_lambda),
                                                      activation='softmax')))

            variable_importance = Input_selection(input)
            final_input = Multiply()([input, variable_importance])

            if len(self.neurons)==1:
                LSTM_layer1=LSTM(self.neurons[0], activation=self.activation, kernel_regularizer=regularizers.l1(self.reg_lambda),
                                    recurrent_regularizer=regularizers.l1(self.reg_lambda))(final_input)
                Last_layer= Dense(1, activation=self.activation, kernel_regularizer=regularizers.l1(self.reg_lambda))(LSTM_layer1)
            else:
                LSTM_layer1 = LSTM(self.neurons[0], activation=self.activation, return_sequences=True,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(final_input)

                LSTM_layer2 = LSTM(self.neurons[0], activation=self.activation,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(LSTM_layer1)
                Last_layer = Dense(1, activation=self.activation, kernel_regularizer=regularizers.l1(self.reg_lambda))(
                    LSTM_layer2)

            lstm_model = Model(input=input, output=Last_layer)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)
            lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

            # fit network
            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                            patience=200, min_lr=0.0001)
                          ]

            lstm_model.fit(trainX, trainY, epochs=self.epoch, batch_size=self.batch, verbose=2,
                      shuffle=False, validation_data=(valX, valY), callbacks=early_stop)

        final_model = load_model(model_path)

        return final_model

    def fittingGRU(self, model_path):

        trainX, valX, testX, trainY, valY, testY, scalerY = self.data_preprocessing()

        if os.path.isfile(model_path):
            final_model = load_model(model_path)
        else:
            input_shape = (self.time_step, self.array_size)
            input = Input(input_shape)
            Input_selection = Sequential()

            for layer_size in self.encoder:
                Input_selection.add(GRU(layer_size, activation='linear', return_sequences=True,
                                         kernel_regularizer=regularizers.l1(self.reg_lambda),
                                         recurrent_regularizer=regularizers.l1(self.reg_lambda),
                                         input_shape=input_shape))

            Input_selection.add(
                TimeDistributed(Dense(self.array_size, kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      activation='softmax')))

            variable_importance = Input_selection(input)
            final_input = Multiply()([input, variable_importance])

            if len(self.neurons) == 1:
                GRU_layer1 = GRU(self.neurons[0], activation=self.activation,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(final_input)
                GRU_layer = Dense(1, activation=self.activation, kernel_regularizer=regularizers.l1(self.reg_lambda))(
                    GRU_layer1)
            else:
                GRU_layer1 = GRU(self.neurons[0], activation=self.activation, return_sequences=True,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(final_input)

                GRU_layer2 = GRU(self.neurons[0], activation=self.activation,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(GRU_layer1)

                GRU_layer = Dense(1, activation=self.activation, kernel_regularizer=regularizers.l1(self.reg_lambda))(
                    GRU_layer2)

            gru_model = Model(input=input, output=GRU_layer)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)
            gru_model.compile(loss='mean_squared_error', optimizer=optimizer)

            # fit network
            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                            patience=200, min_lr=0.0001)
                          ]

            gru_model.fit(trainX, trainY, epochs=self.epoch, batch_size=self.batch, verbose=2,
                           shuffle=False, validation_data=(valX, valY), callbacks=early_stop)

        final_model = load_model(model_path)

        return final_model

    def fittingRNN(self, model_path):

        trainX, valX, testX, trainY, valY, testY, scalerY = self.data_preprocessing()

        if os.path.isfile(model_path):
            final_model = load_model(model_path)
        else:
            input_shape = (self.time_step, self.array_size)
            input = Input(input_shape)
            Input_selection = Sequential()

            for layer_size in self.encoder:
                Input_selection.add(SimpleRNN(layer_size, activation='linear', return_sequences=True,
                                         kernel_regularizer=regularizers.l1(self.reg_lambda),
                                         recurrent_regularizer=regularizers.l1(self.reg_lambda),
                                         input_shape=input_shape))

            Input_selection.add(
                TimeDistributed(Dense(self.array_size, kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      activation='softmax')))

            variable_importance = Input_selection(input)
            final_input = Multiply()([input, variable_importance])

            if len(self.neurons) == 1:
                RNN_layer1 = SimpleRNN(self.neurons[0], activation=self.activation,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(final_input)
                Last_layer = Dense(1, activation=self.activation, kernel_regularizer=regularizers.l1(self.reg_lambda))(
                    RNN_layer1)
            else:
                RNN_layer1 = SimpleRNN(self.neurons[0], activation=self.activation, return_sequences=True,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(final_input)

                RNN_layer2 = SimpleRNN(self.neurons[0], activation=self.activation,
                                   kernel_regularizer=regularizers.l1(self.reg_lambda),
                                   recurrent_regularizer=regularizers.l1(self.reg_lambda))(RNN_layer1)
                Last_layer = Dense(1, activation=self.activation, kernel_regularizer=regularizers.l1(self.reg_lambda))(
                    RNN_layer2)

            rnn_model = Model(input=input, output=Last_layer)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)
            rnn_model.compile(loss='mean_squared_error', optimizer=optimizer)

            # fit network
            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                            patience=200, min_lr=0.0001)
                          ]

            rnn_model.fit(trainX, trainY, epochs=self.epoch, batch_size=self.batch, verbose=2,
                           shuffle=False, validation_data=(valX, valY), callbacks=early_stop)

        final_model = load_model(model_path)

        return final_model

    def predict(self, testX, testY, scalerY, final_model):

        try:
            scaled_pred_Y=final_model.predict(testX)
            if len(scaled_pred_Y.shape)>2:
                scaled_pred_Y = flatten(scaled_pred_Y)
        except:
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1]*testX.shape[2]))
            scaled_pred_Y = final_model.predict(testX)

        if len(testY.shape) > 2:
            testY = flatten(testY)
        pred_Y=scalerY.inverse_transform(scaled_pred_Y)
        true_Y = scalerY.inverse_transform(testY)
        error = pred_Y-true_Y
        error_sq = np.multiply(error, error)

        return pred_Y, true_Y, error_sq

    def evaluation(self, pred_Y, true_Y):

        rmse = []
        mae = []
        mape = []
        r_squared = []
        if pred_Y.shape[1] > 1:
            for i in range(pred_Y.shape[1]):
                rmse.append(math.sqrt(mean_squared_error(true_Y[:, i], pred_Y[:, i])))
                mae.append(mean_absolute_error(true_Y[:, i], pred_Y[:, i]))
                mape.append(np.mean(np.abs((true_Y[:, i] - pred_Y[:, i]) / true_Y[:, i])) * 100)
                r_squared.append(r2_score(true_Y[:, i], pred_Y[:, i]))
        else:
            rmse.append(math.sqrt(mean_squared_error(true_Y, pred_Y)))
            mae.append(mean_absolute_error(true_Y, pred_Y))
            mape.append(np.mean(np.abs((true_Y - pred_Y) / true_Y)) * 100)
            r_squared.append(r2_score(true_Y, pred_Y))

        return rmse, mae, mape, r_squared






