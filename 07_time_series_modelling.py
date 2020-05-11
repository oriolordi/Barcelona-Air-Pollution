# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn import metrics
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet


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


# Naive model predictions
def naive_prediction(data):
    pred = data[-1]
    return pred


# Holt-Winters predictions
def holt_winters_prediction(data, length=1):
    fit = ExponentialSmoothing(np.asarray(data), seasonal_periods=7, trend='add', seasonal='add').fit()
    # Forecast one value in the future
    return fit.forecast(length)


# Sarima predictions
def sarima_prediction(data, pollutant, p, q, length=1):
    seasonality = 7
    if pollutant == 'O3':
        d = 1
    else:
        d = 0
    order_arima = (p, d, q)
    order_sarima = (1, d, 1, seasonality)
    fit = SARIMAX(np.asarray(data), order=order_arima, seasonal_order=order_sarima,
                  initialization='approximate_diffuse').fit()
    # Forecast one value in the future
    return fit.forecast(length)


# Prophet prediction
def prophet_prediction(data, length=1):
    data = data.reset_index()
    data.columns = ['ds', 'y']
    prophet_basic = Prophet()
    prophet_basic.fit(data)
    future = prophet_basic.make_future_dataframe(periods=length)
    forecast = prophet_basic.predict(future)
    yhat = forecast.loc[forecast.index[-(length)], 'yhat']
    return yhat


# Walk-forward validation function
def walk_forward_validation(train, test, model, pollutant):
    keep_going = True
    train_value = train['VALUE']
    test_value = test['VALUE']
    # Log transformation for multiplicative data
    if (model.startswith('Mul')):
        train_value = np.log10(train_value.add(1))
        test_value = np.log10(test_value.add(1))
    history = [x for x in train_value]
    predictions = list()
    if len(model.split('_')) > 1:
        model_name = model.split('_')[1]
    else:
        model_name = model
    for i in range(len(test_value)):
        # Predict
        if model_name == 'Naive':
            yhat = naive_prediction(history)
        elif model_name == 'Holt-Winters':
            yhat = holt_winters_prediction(history, length=1)[0]
        elif model_name == 'Arima AR 2':
            yhat = sarima_prediction(history, pollutant, p=2, q=0, length=1)[0]
        elif model_name == 'Arima MA 2':
            yhat = sarima_prediction(history, pollutant, p=0, q=2, length=1)[0]
        elif model_name == 'Arima AR 2 MA 2':
            yhat = sarima_prediction(history, pollutant, p=2, q=2, length=1)[0]
        elif model_name == 'Prophet':
            yhat = prophet_prediction(train_value, length=1)
        else:
            keep_going = False
            break
        predictions.append(yhat)
        # Observation
        obs = test_value[i]
        history.append(obs)
        train_value.append(test_value[[i]], ignore_index=True)
        #train_value.append(test.iloc[i], ignore_index=True)
        #print('>Predicted={:.2f}, Expected= {:.2f}' .format(yhat, obs))
    if keep_going:
        # Undo log transformation
        if (model.startswith('Mul')):
            predictions = [10**x-1 for x in predictions]
            test_value = 10**test_value
            test_value = test_value.add(-1)
        mse = metrics.mean_squared_error(test_value, predictions)
        rmse = sqrt(mse)
        #print('RMSE: {:.2f}' .format(rmse))
    else:
        rmse = 0
    return rmse, keep_going


# Main function: does the modelling and evaluates the models for each of the available STATION and POLLUTANT data
def model_and_evaluate(dictionary, dictionary_rmse):
    for pollutant, station_df in dictionary.items():
        for station, df in station_df.items():
            # Do the modelling and evaluation only for combinations of STATION and POLLUTANT that have data
            if len(df) > 0:
                # Split train and test sets
                train, test = split_train_test(df)
                # Modelling and predicting for each model (using walk-forward validation)
                for model in model_names:
                    rmse, valid = walk_forward_validation(train, test, model, pollutant)
                    # Save the rmse value to its corresponding place in the dictionary of RMSE values
                    if valid:
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
