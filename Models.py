from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import adam
from keras.models import load_model
from keras.models import Model
import os

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"



class RNN_model():

    def __init__(self, inputs, layers, time_steps,
                 activation, reg_lambda= 0.00001):

        self.inputs = inputs
        self.time_steps = time_steps
        self.layers = layers
        self.activation = activation
        self.reg_lambda = reg_lambda

    def RNN_model(self, act_name):

        input_shape = (self.time_steps, int(self.inputs.shape[2]))

        RNN_MODEL = Sequential(name=act_name)

        if len(self.layers) > 1:
            RNN_MODEL.add(LSTM(self.layers[0], activation=self.activation, return_sequences=True,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))
            for layer in range(1, len(self.layers)):
                if layer < len(self.layers) - 1:
                    RNN_MODEL.add(LSTM(self.layers[layer], activation=self.activation, return_sequences=True,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
                else:
                    RNN_MODEL.add(LSTM(self.layers[layer], activation=self.activation,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            RNN_MODEL.add(LSTM(self.layers[0], activation=self.activation,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))

        rnn_output = RNN_MODEL(self.inputs)

        return rnn_output


class MLP_model():

    def __init__(self, inputs, layers,
                 activation, reg_lambda= 0.00001):

        self.inputs = inputs
        self.layers = layers
        self.activation = activation
        self.reg_lambda = reg_lambda

    def model(self, act_name):

        input_dim = int(self.inputs.shape[1])

        MLP_MODEL = Sequential(name=act_name)

        if len(self.layers) > 1:
            MLP_MODEL.add(Dense(self.layers[0], activation=self.activation,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              input_dim=input_dim))
            for layer in range(1, len(self.layers)):
                if layer < len(self.layers) - 1:
                    MLP_MODEL.add(Dense(self.layers[layer], activation=self.activation,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            MLP_MODEL.add(Dense(self.layers[0], activation=self.activation,
                                kernel_regularizer=regularizers.l1(self.reg_lambda),
                                input_dim=input_dim))

        mlp_output = MLP_MODEL(self.inputs)

        return mlp_output


class Auto_Encoder():

    def __init__(self, trainingX, valX, encoder_layers, decoder_layers,
                 time_steps, activation, batch_size,
                 early_stopping, epoch, lr, model_path, var,
                 reg_lambda= 0.00001):

        self.trainingX = trainingX
        self.valX = valX
        self.time_steps = time_steps
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.array_size = trainingX.shape[2]
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.var = var
        self.epoch = epoch
        self.lr = lr
        self.activation = activation
        self.reg_lambda = reg_lambda

    def encoding(self):

        input_shape = (self.time_steps, self.array_size)
        self.inputs = Input(shape=input_shape)
        encoder = Sequential()

        if len(self.encoder_layers) > 1:
            encoder.add(LSTM(self.encoder_layers[0], activation=self.activation, return_sequences=True,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))

            for layer in range(1, len(self.encoder_layers)):
                if layer < len(self.encoder_layers) - 1:
                    encoder.add(LSTM(self.encoder_layers[layer], activation=self.activation, return_sequences=True,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
                else:
                    encoder.add(LSTM(self.encoder_layers[layer], activation=self.activation,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            encoder.add(LSTM(self.encoder_layers[0], activation=self.activation,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))

        encoder.add(RepeatVector(self.time_steps))

        encoded_output = encoder(self.inputs)

        return encoded_output

    def decoding(self, encoded_output):

        input_shape = (self.time_steps, int(encoded_output.shape[2]))

        decoder = Sequential()

        if len(self.decoder_layers) > 1:

            decoder.add(LSTM(self.decoder_layers[0], activation=self.activation, return_sequences=True,
                             kernel_regularizer=regularizers.l1(self.reg_lambda),
                             recurrent_regularizer=regularizers.l1(self.reg_lambda),
                             input_shape=input_shape))

            for layer in range(1, len(self.decoder_layers)):
                decoder.add(LSTM(self.decoder_layers[layer], activation=self.activation, return_sequences=True,
                                 kernel_regularizer=regularizers.l1(self.reg_lambda),
                                 recurrent_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            decoder.add(LSTM(self.decoder_layers[0], activation=self.activation, return_sequences=True,
                             kernel_regularizer=regularizers.l1(self.reg_lambda),
                             recurrent_regularizer=regularizers.l1(self.reg_lambda),
                             input_shape=input_shape))

        decoder.add(TimeDistributed(Dense(self.array_size)))

        self.decoded_output = decoder(encoded_output)

        return self.decoded_output

    def training(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/model_ae_%s_%s' % (self.var, 1) + '.h5'):

            final_model = load_model('model/model_ae_%s_%s' % (self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingX, epochs=300,
                            batch_size=self.batch_size, verbose=2,
                            shuffle=False, validation_data=(self.valX, self.valX),
                            callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            encoded_output = self.encoding()
            output = self.decoding(encoded_output)

            final_model = Model(self.inputs, output)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingX,  epochs=self.epoch, batch_size=self.batch_size,
                            verbose=2, shuffle=False,
                            validation_data=(self.valX, self.valX), callbacks=early_stop)

        final_model = load_model(self.model_path + '.h5')

        return final_model

    def flatten(self, X):

        flattened_X = np.empty((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]

        return flattened_X

    def reconstruction_error(self, final_model, dataX):

        pred_dataX = final_model.predict(dataX)
        mse_train = np.mean(np.power(self.flatten(dataX) - self.flatten(pred_dataX), 2), axis=1)

        return mse_train



class LSTM_model():

    def __init__(self, trainingX, valX, trainingY, valY,
                 rnn_layers, time_steps, batch_size, activation, output_activation,
                 early_stopping, model_path, method, var, data, epoch, lr, reg_lambda=0.00001):

        self.trainingX = trainingX
        self.valX = valX
        self.trainingY = trainingY
        self.valY = valY
        self.rnn_layers = rnn_layers
        self.time_steps = time_steps
        self.array_size = trainingX.shape[2]
        self.batch_size = batch_size
        self.activation = activation
        self.output_activation = output_activation
        self.epoch = epoch
        self.var = var
        self.data = data
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.lr = lr
        self.method = method
        self.reg_lambda = reg_lambda

    def lstm_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            % ( self.data, self.method,  self.var, self.data, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s_%s_%s'
                            % (self.data, self.method,  self.var, self.data, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                            batch_size=self.batch_size, verbose=2,
                            shuffle=False, validation_data=(self.valX, self.valY),
                            callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            inputs = Input(shape=(self.time_steps, self.array_size,))

            rnn_model = RNN_model(inputs, self.rnn_layers, self.time_steps, self.activation)
            output = rnn_model.RNN_model(self.activation)

            final_output = Dense(1, activation=self.output_activation)(output)

            final_model = Model(inputs=inputs, outputs=final_output)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size,
                               verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model

class AIS_LSTM_model():

    def __init__(self, trainingX, valX, trainingY, valY, weight_layers,
                 rnn_layers, time_steps, batch_size, activation, output_activation,
                 early_stopping, model_path, method, var, data, epoch, lr, reg_lambda=0.00001):

        self.trainingX = trainingX
        self.valX = valX
        self.trainingY = trainingY
        self.valY = valY
        self.rnn_layers = rnn_layers
        self.weight_layers = weight_layers
        self.time_steps = time_steps
        self.array_size = trainingX.shape[2]
        self.batch_size = batch_size
        self.activation = activation
        self.output_activation = output_activation
        self.epoch = epoch
        self.var = var
        self.data = data
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.lr = lr
        self.method = method
        self.reg_lambda = reg_lambda

    def ais_lstm_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s' % (self.data, self.method, self.var, self.data, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s_%s_%s' % (self.data, self.method, self.var, self.data, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                            batch_size=self.batch_size, verbose=2,
                            shuffle=False, validation_data=(self.valX, self.valY),
                            callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            input_shape = (self.time_steps, self.array_size)
            inputs = Input(input_shape)

            Input_selection = Sequential()

            Input_selection.add(LSTM(self.weight_layers[0], activation=self.activation, return_sequences=True,
                                     kernel_regularizer=regularizers.l1(self.reg_lambda),
                                     recurrent_regularizer=regularizers.l1(self.reg_lambda),
                                     input_shape=input_shape))
            if len(self.weight_layers) > 1:
                for layer_size in self.weight_layers[1:]:
                    Input_selection.add(LSTM(layer_size, activation=self.activation, return_sequences=True,
                                             kernel_regularizer=regularizers.l1(self.reg_lambda),
                                             recurrent_regularizer=regularizers.l1(self.reg_lambda)))

            Input_selection.add(TimeDistributed(Dense(self.array_size,
                                                      activation='softmax')))

            variable_importance = Input_selection(inputs)

            importance_model = Model(inputs=inputs, outputs=variable_importance, name='importance_model')

            final_input = Multiply()([inputs, importance_model(inputs)])

            final_input = Flatten()(final_input)

            # rnn_model = RNN_model(final_input, self.rnn_layers, self.time_steps, self.activation)
            # output = rnn_model.RNN_model(self.activation)

            final_output = Dense(1, activation=self.output_activation)(final_input)

            proposed_model = Model(inputs=inputs, outputs=final_output)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size,
                               verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model

class data_preparing():

    def __init__(self, X, Y, time_steps, ratio=0.8):

        self.X = X
        self.Y = Y
        self.time_steps = time_steps
        self.array_size = X.shape[1]
        self.ratio = ratio

    def data_splitting(self, dataX, dataY):

        #### data partitioning
        train_len = int(np.round(len(dataY) * self.ratio, 0))
        trainX, valX = dataX[:train_len], dataX[train_len:]
        trainY, valY = dataY[:train_len], dataY[train_len:]

        return trainX, valX, trainY, valY

    def prepare_timeseries(self):
        dataX, dataY = [], []
        length = len(self.Y) - self.time_steps
        for i in range(0, length):
            if len(self.X.shape)>1:
                dataX.append(self.X[i:(i + self.time_steps), :])
            else:
                dataX.append(self.X[i:(i + self.time_steps)])

            dataY.append(self.Y[(i + self.time_steps)])

        return np.array(dataX), np.array(dataY)

    def data_preprocessing(self):

        dataX, dataY = self.prepare_timeseries()
        trainX, valX, trainY, valY = self.data_splitting(dataX, dataY)

        trainX, valX = np.reshape(trainX, (trainX.shape[0], self.time_steps, self.array_size)), \
                       np.reshape(valX, (valX.shape[0], self.time_steps, self.array_size))

        return trainX, valX, trainY, valY