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
                 early_stopping, epoch, lr, model_path, data, method, var,
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
        self.data = data
        self.method = method
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
                    encoder.add(LSTM(self.encoder_layers[layer], activation='tanh',
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            encoder.add(LSTM(self.encoder_layers[0], activation='tanh',
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))

        encoder.add(RepeatVector(self.time_steps))

        encoded_output = encoder(self.inputs)

        encoder_model = Model(self.inputs, encoded_output)

        return encoded_output, encoder_model

    def decoding(self, encoded_output):

        input_shape = (self.time_steps, int(encoded_output.shape[2]))
        input_decode = Input(shape=input_shape)

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
            decoder.add(LSTM(self.decoder_layers[0], activation='sigmoid', return_sequences=True,
                             kernel_regularizer=regularizers.l1(self.reg_lambda),
                             recurrent_regularizer=regularizers.l1(self.reg_lambda),
                             input_shape=input_shape))

        decoder.add(TimeDistributed(Dense(self.array_size)))

        decoded_output = decoder(input_decode)

        decoder_model = Model(input_decode, decoded_output, name="AutoEncoder_loss")


        return decoded_output, decoder_model

    def training(self):

        if os.path.isfile(self.model_path + '_encoder.h5'):

            encoder=load_model(self.model_path + '_encoder.h5')
            decoder=load_model(self.model_path + '_decoder.h5')

            return encoder, decoder


        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            % (self.data, self.method, self.var, self.data, 1) + '_ae.h5'):

            final_model = load_model('model/model_ae_%s_%s' % (self.var, 1) + '_ae.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '_ae.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingX, epochs=300,
                            batch_size=self.batch_size, verbose=2,
                            shuffle=False, validation_data=(self.valX, self.valX),
                            callbacks=early_stop)


            encoder = load_model(self.model_path + '_encoder.h5')
            decoder = load_model(self.model_path + '_decoder.h5')

            return encoder, decoder

        else:

            encoded_output, encoder = self.encoding()
            output, decoder = self.decoding(encoded_output)

            outputs = decoder(encoder(self.inputs))
            final_model = Model(self.inputs, outputs, name = 'AutoEncoder_loss')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '_ae.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingX,  epochs=self.epoch, batch_size=self.batch_size,
                            verbose=2, shuffle=False,
                            validation_data=(self.valX, self.valX), callbacks=early_stop)

            encoder.save(self.model_path + '_encoder.h5')
            decoder.save(self.model_path + '_decoder.h5')

            return encoder, decoder

    def flatten(self, X):

        flattened_X = np.empty((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]

        return flattened_X

    def reconstruction_error(self, final_model, dataX):

        pred_dataX = final_model.predict(dataX)
        mse_train = np.mean(np.power(self.flatten(dataX) - self.flatten(pred_dataX), 2), axis=1)

        return mse_train



class LSTM_model():  #simple LSTM model

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

class AIS_LSTM_model():  #our proposed AIS-LSTM model

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

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            % (self.data, self.method, self.var, self.data, 1) + '.h5'):

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

class Uber_LSTM(): #Uber LSTM model

    def __init__(self, trainingX, valX, trainingY, valY,
                 encoding_layers, decoding_layers, rnn_layers, activation,
                 time_steps, loss_weight, batch_size, early_stopping,
                 model_path, data, var, method, epoch, lr, reg_lambda=0.00001):

        self.trainingX = trainingX
        self.valX = valX
        self.trainingY = trainingY
        self.valY = valY
        self.encoding_layers = encoding_layers
        self.decoding_layers = decoding_layers
        self.rnn_layers = rnn_layers
        self.activation = activation
        self.time_steps = time_steps
        self.array_size = trainingX.shape[2]
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.epoch = epoch
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.data = data
        self.var = var
        self.method = method
        self.lr = lr
        self.reg_lambda = reg_lambda

    def Uber_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            % (self.data, self.method, self.var, self.data, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s_%s_%s'
                            % (self.data, self.method, self.var, self.data, 1) + '.h5')

            losses = {'AutoEncoder_loss': 'mean_squared_error',
                      'Prediction_loss': 'mean_squared_error'}

            lossWeights = {"AutoEncoder_loss": self.loss_weight, "Prediction_loss": 1 - self.loss_weight}

            optimizer = adam(lr=self.lr, epsilon=None, decay=0, amsgrad=False)
            final_model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_Prediction_loss_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_Prediction_loss_loss',
                                          verbose=1,
                                          save_best_only=True)]

            final_model.fit([self.trainingX, self.trainingX],
                            [self.trainingX, self.trainingY], epochs=300,
                            batch_size=self.batch_size, verbose=2,
                            shuffle=False,
                            validation_data=([self.valX, self.valX], [self.valX, self.valY]),
                            callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            input_ae = Input(shape=(self.time_steps, self.array_size), name='ae_input')
            input_lstm = Input(shape=(self.time_steps, self.array_size), name='conc_input')

            ae = Auto_Encoder(trainingX=self.trainingX, valX=self.valX, encoder_layers=[32, 16, 8],
                              decoder_layers=[16, 32], time_steps=self.time_steps,
                              activation=self.activation,
                              batch_size=self.batch_size, early_stopping=self.early_stopping,
                              epoch=self.epoch, lr=self.lr,
                              model_path=self.model_path, data=self.data, method=self.method,
                              var=self.var, reg_lambda=0.00001)

            encoder, decoder = ae.training()

            auto_encoder_output = decoder(encoder(input_ae))

            rnn_input = Concatenate()([input_lstm, encoder(input_ae)])

            inputs = Input(shape=(self.time_steps, int(rnn_input.shape[2])), name='rnn_input')

            rnn_model = RNN_model(inputs, self.rnn_layers,
                                  self.time_steps, self.activation)

            rnn_output = rnn_model.RNN_model(self.activation)

            final_output = Dense(1, activation=self.activation)(rnn_output)

            prediction_model = Model(inputs, final_output, name="Prediction_loss")

            prediction_output=prediction_model(rnn_input)

            uber_lstm_model = Model(inputs=[input_ae, input_lstm],
                                outputs=[auto_encoder_output, prediction_output])

            losses = {'AutoEncoder_loss': 'mean_squared_error',
                      'Prediction_loss': 'mean_squared_error'}

            lossWeights = {"AutoEncoder_loss": self.loss_weight, "Prediction_loss": 1-self.loss_weight}

            optimizer = adam(lr=self.lr, epsilon=None, decay=0, amsgrad=False)
            uber_lstm_model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_Prediction_loss_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_Prediction_loss_loss',
                                          verbose=1,
                                          save_best_only=True)]
            uber_lstm_model.fit([self.trainingX, self.trainingX],
                            [self.trainingX, self.trainingY], epochs=self.epoch,
                            batch_size=self.batch_size, verbose=2,
                            shuffle=False,
                            validation_data=([self.valX, self.valX],
                                             [self.valX, self.valY]),
                            callbacks=early_stop)

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