"""
    This script is used to train GRU models.
    To run this script, 2 parameters are needed. First is the ticker you want to train.
    Second is the feature you want to use.
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
from keras.callbacks import ModelCheckpoint
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU



# ticker = 'AAPL'
# features = ['Open','High','Low','Close','AdjClose','Volume',ticker]
ticker = sys.argv[1]
features = eval(sys.argv[2])
pre_days = 1
number_of_training = 2300

def stock_price_GRU_pre_processing(df,memory_his_days,pre_days=pre_days):
    pd.set_option('display.max_columns', None)
    df.dropna(inplace=True)
    df['Y'] = df['Close'].shift(-pre_days)
    # standardize
    scalar_X = StandardScaler()
    sca_X = scalar_X.fit_transform(df[features].values)
    # construct feature spaces
    X = []
    deq = deque(maxlen=memory_his_days)
    for data in sca_X:
        deq.append(list(data))
        if len(deq) == memory_his_days:
            X.append(list(deq))
    # last data is not enough to construct the training set

    # discard tail due to lack of Y, preserve discarded data in variable x_last
    X_last = sca_X[-pre_days:, ]
    X = np.array(X[:-pre_days])

    # construct Y
    Y = df['Y'][memory_his_days - 1:-pre_days]
    number_of_instance = len(Y)
    Y = np.array(Y)
    # train test split
    X_train = X[:number_of_training, :, :]
    X_test = X[number_of_training:, :, :]
    Y_train = Y[:number_of_training]
    Y_test = Y[number_of_training:]
    return scalar_X, X_train, X_test, Y_train, Y_test


memory_his_days_list = [1,2,3,4,5,6,7]
number_of_GRU_layers = [1,2,3]
number_of_dense = [1,2,3]
unit_per_layer = [128,256,512]
activation = 'relu'

for   current_GRU_layer_number in   number_of_GRU_layers:
    for current_dense_number  in number_of_dense:
        for  current_unit_per_layer in  unit_per_layer:
            for current_window in memory_his_days_list:
                if features[-1] == ticker:
                    checkpoint_filepath = '../'+ticker+'/trend/{val_mape:.2f}_{epoch:02d}_' +f'window_length_{current_window}_gru_layers_{current_GRU_layer_number}_dense_{current_dense_number}_unit_{current_unit_per_layer}'
                else:
                    checkpoint_filepath = '../'+ticker+'/without_trend/{val_mape:.2f}_{epoch:02d}_' +f'window_length_{current_window}_gru_layers_{current_GRU_layer_number}_dense_{current_dense_number}_unit_{current_unit_per_layer}'
                model_checkpoint_callback = ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=False,
                    monitor='val_mape',
                    mode='min',
                    save_best_only=True)
                scalar_X, X_train, X_test, Y_train, Y_test = stock_price_GRU_pre_processing(
                    pd.read_csv('../../../Feature_Engineering/result/' + ticker + '.csv').drop('Unnamed: 0', axis=1),
                    current_window, pre_days)

                # build the model
                model = Sequential()
                model.add(GRU(current_unit_per_layer, input_shape=X_train.shape[1:], activation=activation, return_sequences=True))
                model.add(Dropout(0.1))

                for i in range(current_GRU_layer_number):
                    model.add(GRU(current_unit_per_layer, activation=activation, return_sequences=True))
                    model.add(Dropout(0.5))

                for j in range(current_dense_number):
                    model.add(Dense(current_unit_per_layer, activation=activation))
                    model.add(Dropout(0.1))
                model.add(Dense(1))
                # compile
                model.compile(optimizer='adam',
                              loss='mse',
                              metrics=['mape'])

                model.fit(X_train,Y_train,batch_size=100,epochs=50,validation_data=(X_test,Y_test),callbacks=[model_checkpoint_callback])
