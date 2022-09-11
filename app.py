# import data preprocessing/visualization libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
from pandas import read_csv
import math

# import keras libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dense, Activation, Dropout

# import tensorflow libraries
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# load in data
df2 = pd.read_csv('TSLA.csv'), parse_dates=['Date'], index_col='Date')
df = df2[['Adj Close', 'Volume']]

def scale_data(df):
    """
    Accepts a pandas dataframe, scales the columns according to
    the minimum and maximum values then stores the scalers and scaled
    columns with their respective unscaled columns in the proper dictionaries.
    The function returns the scaled pandas dataframe, and the dictionary of
    scalers.
    """
    
    
    scaler_dict = {}
    scaled_data = {}

    for col in df.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        feat = df[col].values.reshape(-1, 1)
        col_scaled = scaler.fit_transform(feat)
        scaled_data[col] = col_scaled.flatten()
        scaler_dict[col] = scaler
    df_scaled = pd.DataFrame.from_dict(scaled_data)

    return df_scaled, scaler_dict

def create_dataset(data, look_back=None, look_ahead=None, predict_only_last=None):
    """
    Accepts a 2d array and returns X and y sequences containing 
    look_back + look_ahead + 1 timesteps information for a total of
    (time_series_size) - (look_back + look_ahead + 1) elements,
    each element of X being of look_back length; and each of y 
    either length look_ahead or length one if predict_only_last.
    """


    X_data, Y_data = [], []
    n_samples = len(data)
    window = look_back + look_ahead
    n_sequences = n_samples - window + 1
    print(f'Generating {n_sequences} X, Y samples')

    if data.shape[1] > 1:
        y_data = data[:, 0]
    else:
        y_data = data

    for i in range(n_sequences):
        x = data[i: i+look_back]
        y = y_data[i+look_back : i + look_back + look_ahead]

        if predict_only_last:
            y = y[-1]
        X_data += [x]
        Y_data += [y]

    return np.array(X_data), np.array(Y_data)
  
def create_split(df, look_back=None, look_ahead=None, train_size=0.70, predict_only_last=None):
    """
    Returns a training and testing split for the 2d array that is passed in.
    """
    
    
    n_samples = df.shape[0]
    train_size = int(n_samples * train_size)
    train = df.iloc[:train_size].values
    test = df.iloc[train_size:].values
    X_train, y_train = create_dataset(train, look_back=look_back,
                                      look_ahead=look_ahead,
                                      predict_only_last=predict_only_last)
    X_test, y_test = create_dataset(test, look_back=look_back,
                                    look_ahead=look_ahead,
                                    predict_only_last=predict_only_last)

    return X_train, y_train, X_test, y_test

def inverse_scaling(data, scaler_dict, output_feat_name):
    """
    Accepts still-scaled forecast sequences and returns
    the real-time forecasts.
    """
    
    
    return scaler_dict[output_feat_name].inverse_transform(data)
    
df, scalers = scale_data(df)
look_back, look_ahead = 28, 7
predict_only_last = False
X_train, y_train, X_test, y_test = create_split(df, look_back=look_back,
                                                look_ahead=look_ahead,
                                                train_size=0.7,
                                                predict_only_last=predict_only_last)

# set the epochs and dimensions of data
n_feats = 2
epochs = 25
batch_size = 32
dropout_prob = .5
input_shape = (look_back, n_feats)

# initialize sequential model
opt = optimizers.Nadam(learning_rate=.01)
model = Sequential()
model.add(LSTM(256, input_shape=input_shape, activation='tanh', return_sequences=False))
# dropout regularization
model.add(Dropout(dropout_prob))
model.add(Dense(look_ahead, activation='relu'))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
model.summary()

# fit the model with learning rate scheduler callback
lr_scheduler = LearningRateScheduler(
    lambda epoch: 1e-4*10**(epoch/10))
schedule_results = model.fit(X_train, y_train,
                             epochs=40, batch_size=batch_size,
                             verbose=1,
                             callbacks=[lr_scheduler])
                             
# plot the loss to determine appropriate learning_rate
plt.semilogx(schedule_results.history['lr'], schedule_results.history['loss'])
plt.axis((1.e-4, 1.e-1, 0, np.max(schedule_results.history['loss'])))
plt.xlabel('learning rate')
plt.ylabel('loss')
plt.grid()

# now rebuild the sequential model with the optimal learning_rate
new_lr = 1.e-3
opt = optimizers.Nadam(learning_rate=new_lr)
model = Sequential()
model.add(LSTM(256, input_shape=input_shape, activation='tanh', return_sequences=False))
# dropout regularization
model.add(Dropout(dropout_prob))
model.add(Dense(look_ahead, activation='relu'))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
model.summary()

# retrain one last time with early stopping and a patience of 10
early_stopping = EarlyStopping(monitor='loss', patience=10,
                               min_delta=1.e-6)
history = model.fit(X_train, y_train,
                    epochs=100, batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks = [early_stopping])
                    
# retrieve the still-scaled predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# unscale the data
y_train = inverse_scaling(y_train, scalers, 'Adj Close')
y_test = inverse_scaling(y_test, scalers, 'Adj Close')

train_predict = inverse_scaling(train_predict, scalers, 'Adj Close')
test_predict = inverse_scaling(test_predict, scalers, 'Adj Close')

# this sequence contains the forecasted values for look_ahead timesteps
seven_day_prediction = test_predict[-1]
