# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn import metrics
from math import sqrt
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


# Decision Tree (DT) prediction function
def supervised_prediction(training, value_to_predict, model_name, keep_going):
    if model_name == 'DT':
        model = DecisionTreeRegressor()
    elif model_name == 'KNN':
        model = KNeighborsRegressor()
    elif model_name == 'RF':
        model = RandomForestRegressor(n_estimators=100)
    else:
        keep_going = False
    if keep_going:
        model.fit(pd.DataFrame(training['VALUE']), training['TARGET'])
        yhat = model.predict(pd.DataFrame({'VALUE_TO_PREDICT': value_to_predict}, index={0}))[0]
    else:
        yhat = 0
    return yhat, keep_going


# Walk-forward validation function
def walk_forward_validation(train, test, model):
    keep_going = True
    # Change data to a supervised learning problem
    train_data = pd.DataFrame(train.reset_index()['VALUE'])
    test_data = pd.DataFrame(test.reset_index()['VALUE'])
    train_data = train_data.assign(TARGET=train_data['VALUE'].shift(-1))
    training = train_data.loc[train_data.index[:-1], ['VALUE', 'TARGET']] #df
    value_to_predict = train_data.loc[train_data.index[-1], 'VALUE'] #float
    predictions = list()
    for i in range(len(test_data)):
        yhat, keep_going = supervised_prediction(training, value_to_predict, model, keep_going)
        if not keep_going:
            break
        predictions.append(yhat)
        training = training.append({'VALUE': value_to_predict, 'TARGET': yhat}, ignore_index=True)
        value_to_predict = test_data.iloc[i].values[0]
        # print('>Predicted={:.2f}, Expected= {:.2f}' .format(yhat, value_to_predict))
    if keep_going:
        mse = metrics.mean_squared_error(test_data, predictions)
        rmse = sqrt(mse)
        # print('RMSE: {:.2f}' .format(rmse))
        # print('test data')
        # print(list(test_data['VALUE']))
        # print('predictions')
        # print(predictions)
    else:
        rmse = 0
    return rmse, keep_going


# Main function: does the modelling and evaluates the models for each of the available STATION and POLLUTANT data
def model_and_evaluate(dictionary, dictionary_rmse):
    n_repeats = 10
    for pollutant, station_df in dictionary.items():
        for station, df in station_df.items():
            # Do the modelling and evaluation only for combinations of STATION and POLLUTANT that have data
            if len(df) > 0:
                # Split train and test sets
                train, test = split_train_test(df)
                # Modelling and predicting for each model (using walk-forward validation)
                for model in model_names:
                    if model.startswith('Supervised'):
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
