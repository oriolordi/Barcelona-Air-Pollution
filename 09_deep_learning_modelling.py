# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn import metrics
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import ConvLSTM2D
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# Turn off automatic plot showing
plt.ioff()


# Load the datasets (daily average and daily maximum value of pollutants in each station) + weather data + RMSE values
df_mean = pd.read_csv('datasets/pollution_mean.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_max = pd.read_csv('datasets/pollution_max.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_weather = pd.read_csv('datasets/weather_data.csv', parse_dates=[0], index_col=[0])
df_rmse = pd.read_csv('datasets/rmse.csv', index_col=[0])
df_rmse_mean = pd.read_csv('datasets/pollution_rmse_mean.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_rmse_max = pd.read_csv('datasets/pollution_rmse_max.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
# Make a copy of the dataset just in case
df_mean_vault = df_mean.copy()
df_max_vault = df_max.copy()
df_weather_vault = df_weather.copy()
df_rmse_vault = df_rmse.copy()
df_rmse_mean_vault = df_rmse_mean.copy()
df_rmse_max_vault = df_rmse_max.copy()


# Function to save a dictionary of dictionaries for each pollutant and each station with the dataframe of values
# while also merging the weather information.
# also, save Februrary of 2020 as validation data for later use
def convert_to_dict(df):
    dict_df = {key: {} for key in df['POLLUTANT'].unique()}
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            df_station_pollutant = df['VALUE'][(df['STATION'] == station) & (df['POLLUTANT'] == pollutant)]
            df_merged = pd.merge(df_station_pollutant, df_weather, how='inner', left_index=True, right_index=True)
            dict_df[pollutant].update({station: df_merged})
    return dict_df


# Convert the dataframe to dictionaries
dict_df_mean = convert_to_dict(df_mean)
dict_df_max = convert_to_dict(df_max)


# Function to convert the RMSE dataframe to a dictionary
def convert_rmse_dict(df):
    dict_df = {key: {} for key in df['POLLUTANT'].unique()}
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            df_specific = df['RMSE'][(df['STATION'] == station) & (df['POLLUTANT'] == pollutant)]
            dict_df[pollutant].update({station: df_specific})
    return dict_df


# Convert RMSE dataframe to a dictionary
dict_rmse = convert_rmse_dict(df_rmse)
dict_rmse_mean = convert_rmse_dict(df_rmse_mean)
dict_rmse_max = convert_rmse_dict(df_rmse_max)


# Function to get the model names from the dictionary of RMSE values
def get_model_names(dictionary):
    model_names_func = []
    for i in dictionary.values():
        for j in i.values():
            model_names_func.append(list(j.index))
    model_names_func = model_names_func[0]
    return model_names_func


# Get the model names from the dictionary of RMSE values
model_names = get_model_names(dict_rmse)
model_names_mean = get_model_names(dict_rmse_mean)
model_names_max = get_model_names(dict_rmse_max)
existing_model_names_mean = set(model_names).intersection(model_names_mean)
existing_model_names_max = set(model_names).intersection(model_names_max)


# Function to fill the dict_rmse with the existing values previously saved in dict_rmse_mean and dict_rmse_max
def fill_dict_rmse(dictionary_rmse, dictionary_rmse_existing, meanmax):
    if meanmax == 'mean':
        existing_model_names = existing_model_names_mean
    elif meanmax == 'max':
        existing_model_names = existing_model_names_max
    for pollutant, station_df in dictionary_rmse_existing.items():
        for station, df in station_df.items():
            for model in existing_model_names:
                dictionary_rmse[pollutant][station][model] = dictionary_rmse_existing[pollutant][station][model]
    return dictionary_rmse


# Fill the dict_rmse with the preexisting values
dict_rmse_mean = fill_dict_rmse(copy.deepcopy(dict_rmse), dict_rmse_mean, 'mean')
dict_rmse_max = fill_dict_rmse(copy.deepcopy(dict_rmse), dict_rmse_max, 'max')


# Train-test splitting function
def split_train_test(df):
    train = df[np.logical_not((df.index.month == 3) & (df.index.year == 2020))]
    test = df[(df.index.month == 3) & (df.index.year == 2020)]
    return train, test


# Fit the MultiLayer Perceptron (MLP) model
def model_mlp_fit(train, config):
    # unpack config
    n_input, n_nodes, n_epochs, n_batch = config
    # prepare data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    # define model
    model = Sequential()
    model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# Forecast with a MLP model
def model_mlp_predict(model, history, config):
    # unpack config
    n_input, _, _, _ = config
    # prepare data
    x_input = np.array(history[-n_input:]).reshape(1, n_input)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


# Fit a Convolutional Neural Network (CNN) model
def model_cnn_fit(train, config):
    # unpack config
    n_input, n_filters, n_kernel, n_epochs, n_batch = config
    # prepare data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    # define model
    model = Sequential()
    model.add(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(n_input, 1)))
    model.add(Conv1D(n_filters, n_kernel, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# Forecast with a CNN model
def model_cnn_predict(model, history, config):
    # unpack config
    n_input, _, _, _, _ = config
    # prepare data
    x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


# Difference dataset
def difference(data, interval):
    return [data[i] - data[i - interval] for i in range(interval, len(data))]


# Fit a Long Short-Term Memory (LSTM) model
def model_lstm_fit(train, config):
    # unpack config
    n_input, n_nodes, n_epochs, n_batch, n_diff = config
    # prepare data
    if n_diff > 0:
        train = difference(train, n_diff)
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# Forecast with a LSTM model
def model_lstm_predict(model, history, config):
    # unpack config
    n_input, _, _, _, n_diff = config
    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return correction + yhat[0]


# Fit a Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM) model
def model_cnn_lstm_fit(train, config):
    # unpack config
    n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
    n_input = n_seq * n_steps
    # prepare data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, 1))
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None,n_steps,1))))
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_nodes, activation='relu'))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# Forecast with a CNN-LSTM model
def model_cnn_lstm_predict(model, history, config):
    # unpack config
    n_seq, n_steps, _, _, _, _, _ = config
    n_input = n_seq * n_steps
    # prepare data
    x_input = np.array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


# Fit Convolutional LSTM (ConvLSTM) model
def model_convlstm_fit(train, config):
    # unpack config
    n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
    n_input = n_seq * n_steps
    # prepare data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], n_seq, 1, n_steps, 1))
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(n_filters, (1,n_kernel), activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
    model.add(Flatten())
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# Forecast with a ConvLSTM model
def model_convlstm_predict(model, history, config):
    # unpack config
    n_seq, n_steps, _, _, _, _, _ = config
    n_input = n_seq * n_steps
    # prepare data
    x_input = np.array(history[-n_input:]).reshape((1, n_seq, 1, n_steps, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


# Fit Supervised Learning model
def model_supervised(train, model_name, cfg):
    if model_name == 'DT':
        model = DecisionTreeRegressor()
    elif model_name == 'KNN':
        model = KNeighborsRegressor()
    elif model_name == 'RF':
        model = RandomForestRegressor(n_estimators=100)
    n_input = cfg
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    model.fit(train_x, train_y)
    return model

# Forecast with a Supervised Learning model
def model_supervised_predict(model, history, config):
    # unpack config
    n_input = config
    # prepare data
    x_input = np.array(history[-n_input:]).reshape(1, n_input)
    # forecast
    yhat = model.predict(x_input)
    return yhat[0]


# Transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)t
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values


# Walk-forward validation function
def walk_forward_validation(train, test, model_name):
    predictions = list()
    keep_going = True
    # fit model
    if model_name == 'MLP':
        cfg = [14, 500, 100, 100]
        model = model_mlp_fit(train, cfg)
    elif model_name == 'CNN':
        cfg = [14, 256, 2, 100, 100]
        model = model_cnn_fit(train, cfg)
    elif model_name == 'LSTM':
        cfg = [14, 50, 100, 100, 7]
        model = model_lstm_fit(train, cfg)
    elif model_name == 'CNN-LSTM':
        cfg = [2, 7, 64, 2, 100, 200, 100]
        model = model_cnn_lstm_fit(train, cfg)
    elif model_name == 'ConvLSTM':
        cfg = [2, 7, 256, 2, 200, 200, 100]
        model = model_convlstm_fit(train, cfg)
    elif model_name == 'DT' or model_name == 'KNN' or model_name == 'RF':
        cfg = 7
        model = model_supervised(train, model_name, cfg)
    else:
        keep_going = False
    if keep_going:
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            if model_name == 'MLP':
                yhat = model_mlp_predict(model, history, cfg)
            elif model_name == 'CNN':
                yhat = model_cnn_predict(model, history, cfg)
            elif model_name == 'LSTM':
                yhat = model_lstm_predict(model, history, cfg)
            elif model_name == 'CNN-LSTM':
                yhat = model_cnn_lstm_predict(model, history, cfg)
            elif model_name == 'ConvLSTM':
                yhat = model_convlstm_predict(model, history, cfg)
            elif model_name == 'DT' or model_name == 'KNN' or model_name == 'RF':
                yhat = model_supervised_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        rmse = sqrt(metrics.mean_squared_error(test, predictions))
    else:
        rmse = 0
    return rmse, keep_going


# Main function: does the modelling and evaluates the models for each of the available STATION and POLLUTANT data
def model_and_evaluate(dictionary, dictionary_rmse):
    k = 0
    n_repeats = 10
    for pollutant, station_df in dictionary.items():
        for station, df in station_df.items():
            # Do the modelling and evaluation only for combinations of STATION and POLLUTANT that have data
            if len(df) > 0:
                k = k + 1
                print(k)
                # Split train and test sets
                train, test = split_train_test(df)
                train = train['VALUE']
                test = test['VALUE']
                # Modelling and predicting for each model (using walk-forward validation)
                for model in model_names:
                    if model.startswith('DeepLearning') or model.startswith('Super7'):
                        # Perform the walk forward validation n_repeats times (each time returning the RMSE)
                        results = [walk_forward_validation(train, test, model.split(' ')[1]) for _ in range(n_repeats)]
                        # Save the mean RMSE value to its corresponding place in the dictionary of RMSE values
                        if results[0][1]:  # keep_going == True
                            rmse = np.mean([result[0] for result in results])
                            dictionary_rmse[pollutant][station][model] = rmse
    return dictionary_rmse


# Model and evaluate the data using the mean and the max values of the pollutants
# the goal of this function is basically to fill  values of the dictionaries (dict_rmse_mean and dict_rmse_max)
# these dictionaries contain for each POLLUTANT and each STATION a dataframe with a column of RMSE values for each model
dict_rmse_mean = model_and_evaluate(dict_df_mean, dict_rmse_mean)
dict_rmse_max = model_and_evaluate(dict_df_max, dict_rmse_max)


# Function to convert the dictionary of dictionaries with RMSE values back to a dataframe
def convert_back_to_df(dictionary):
    df_rmse_list = []
    for pollutant, station_df in dictionary.items():
        for station, df in station_df.items():
            df_to_append = pd.DataFrame(df)
            df_to_append = df_to_append.assign(STATION=station, POLLUTANT=pollutant)
            df_rmse_list.append(df_to_append)
    return pd.concat(df_rmse_list)


# Convert back the RMSE mean and max dictionaries to a dataframe
df_rmse_mean = convert_back_to_df(dict_rmse_mean)
df_rmse_max = convert_back_to_df(dict_rmse_max)


# Reorder dataframe columns
order = ['POLLUTANT', 'STATION', 'RMSE']
df_rmse_mean = df_rmse_mean[order]
df_rmse_max = df_rmse_max[order]


# Save the preprocessed csv files
df_rmse_mean.to_csv('datasets/pollution_rmse_mean.csv')
df_rmse_max.to_csv('datasets/pollution_rmse_max.csv')
# Lines to correctly read this csv (for future use)
# df_rmse_mean = pd.read_csv('datasets/pollution_rmse_mean.csv', parse_dates=[0], index_col=[0],
#                  dtype={'STATION': 'category', 'POLLUTANT': 'category'})
# df_rmse_max = pd.read_csv('datasets/pollution_rmse_max.csv', parse_dates=[0], index_col=[0],
#                  dtype={'STATION': 'category', 'POLLUTANT': 'category'})
