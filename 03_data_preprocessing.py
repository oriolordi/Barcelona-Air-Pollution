# Import statements
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import copy


# Turn off automatic plot showing
plt.ioff()


# Load the dataset
df = pd.read_csv('datasets/pollution_missing_dates.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
# Make a copy of the dataset just in case
df_vault = df.copy()


# Save a dictionary of dictionaries for each pollutant and each station with the dataframe of values
dict_df = {key: {} for key in df['POLLUTANT'].unique()}
for pollutant in df['POLLUTANT'].unique():
    for station in df['STATION'].unique():
        df_station_pollutant = df['VALUE'][(df['STATION'] == station) & (df['POLLUTANT'] == pollutant)]
        dict_df[pollutant].update({station: df_station_pollutant})


# Cheating a bit:
# since the data is displayed from 1 hour to 24 hours for each day, but python interprets midnight as the next day,
# all hours will be delayed one hour so that when the samples are resampled to a daily frequency,
# the 24 hours taken for the resampling are the ones corresponding to that day.
# OBSERVATION: this is only valid because of the daily resampling, if the data was to be analyzed hourly,
# it would make no sense to delay observations by one hour (or it would, but then it would have to be taken into account
# when predicting future values)

# Interpolate missing values and resample to a daily frequency of each dataframe
dict_df_mean = copy.deepcopy(dict_df)
dict_df_max = copy.deepcopy(dict_df)
for pollutant, station_df in dict_df.items():
    for station, df in station_df.items():
        if len(df) > 0:
            # Delay one hour
            df.index = df.index - timedelta(hours=1)
            # Interpolate missing values
            df = df.interpolate(method='linear')
            # Update values
            dict_df[pollutant].update({station: df})
            # Resample to the daily average and update values
            df_mean = df.resample('D').mean().reset_index().set_index('index')
            dict_df_mean[pollutant].update({station: df_mean})
            # Resample to th daily maximum and update values
            df_max = df.resample('D').max().reset_index().set_index('index')
            dict_df_max[pollutant].update({station: df_max})


# Function to plot the time series of each pollutant in each station
def plot_time_series(dictionary, type):
    dictionary_plot = copy.deepcopy(dictionary)
    for pollutant, station_df in dictionary_plot.items():
        plt.figure()
        for i, (station, df) in enumerate(station_df.items()):
            g = plt.subplot(len(dict_df), 1, i + 1)
            if len(df) == 0:
                idx = pd.date_range(min(df_vault.index), max(df_vault.index), freq='1H')
                df = df.reindex(idx, fill_value=-1)
                plt.ylim([0, 1])
                plt.yticks([0, 1])
            plt.plot(df)
            plt.title(station, y=0)
            if i < (len(station_df) - 1):
                plt.xticks([])
        plt.suptitle('{} time series \n \n Pollutant = {}' .format(type, pollutant), fontweight='bold')
        plt.show()


# Plot the time series for hourly, daily average and daily max values
plot_time_series(dict_df, 'Hourly (interpolated)')
plot_time_series(dict_df_mean, 'Daily average')
plot_time_series(dict_df_max, 'Daily maximum')


# Combinations of POLLUTANT and STATION that need outlier removing:
# Daily MEAN values:
outliers_mean = [['NOx', 'Horta-Guinardo'], ['NO', 'Horta-Guinardo'], ['SO2', 'Gracia'], ['CO', 'Les Corts'],
                 ['PM10', 'Horta-Guinardo']]


# Daily MAX values:
outliers_max = [['NOx', 'Horta-Guinardo'], ['NO', 'Horta-Guinardo'], ['SO2', 'Gracia'], ['CO', 'Les Corts'],
                 ['PM10', 'Horta-Guinardo'], ['SO2', 'Eixample'], ['PM10', 'Eixample']]


# Function to eliminate outliers from the needed time series
def treat_outliers(df_out):
    # Replace the values that deviate more than 15 times the interquartile range by the max value of the series without
    # those values
    array = df_out['VALUE'].values
    iqr = np.percentile(array, 75) - np.percentile(array, 25)
    array[array > (15 * iqr)] = -1
    array[array == -1] = array.max()
    df_out.loc[:, 'VALUE'] = array
    return df_out


# Function to detect which combination of POLLUTANT and STATION (which time series) needs outlier treatment
def detect_and_treat_outliers(dictionary, outliers):
    for pollutant, station_df in dictionary.items():
        for i, (station, df) in enumerate(station_df.items()):
            for outlier in outliers:
                if (pollutant == outlier[0]) & (station == outlier[1]):
                    df_outliers = treat_outliers(df)
                    dictionary[pollutant].update({station: df_outliers})
    return dictionary


# Eliminate outliers from the needed time series
dict_df_mean_outliers = copy.deepcopy(dict_df_mean)
dict_df_max_outliers = copy.deepcopy(dict_df_max)
dict_df_mean_outliers = detect_and_treat_outliers(dict_df_mean_outliers, outliers_mean)
dict_df_max_outliers = detect_and_treat_outliers(dict_df_max_outliers, outliers_max)


# Plot the time series again (now with removed outliers to make sure the outliers have been correctly removed)
plot_time_series(dict_df_mean_outliers, 'Daily average (without outliers)')
plot_time_series(dict_df_max_outliers, 'Daily maximum (without outliers)')


# Check the outliers were corrected properly
for outlier in outliers_mean:
    print(dict_df_mean[outlier[0]][outlier[1]].sort_values(by='VALUE'))
    print(dict_df_mean_outliers[outlier[0]][outlier[1]].sort_values(by='VALUE'))


# Function to convert the dictionary of dictionaries with interpolated and/or resampled values back to a dataframe
def convert_back_to_df(dictionary):
    df_preprocessed_list = []
    for pollutant, station_df in dictionary.items():
        for station, df in station_df.items():
            df_to_append = pd.DataFrame(df.copy())
            df_to_append['STATION'] = station
            df_to_append['POLLUTANT'] = pollutant
            df_preprocessed_list.append(df_to_append)
    return pd.concat(df_preprocessed_list)


# Convert back the time series for hourly, daily average and daily max values to a dataframe
df_hourly = convert_back_to_df(dict_df)
df_mean = convert_back_to_df(dict_df_mean_outliers)
df_max = convert_back_to_df(dict_df_max_outliers)


# Reorder dataframe columns
order = ['POLLUTANT', 'STATION', 'VALUE']
df_hourly = df_hourly[order]
df_mean = df_mean[order]
df_max = df_max[order]


# Save the preprocessed csv files
df_hourly.to_csv('datasets/pollution_hourly.csv')
df_mean.to_csv('datasets/pollution_mean.csv')
df_max.to_csv('datasets/pollution_max.csv')
# Lines to correctly read this csv (for future use)
#df_hourly = pd.read_csv('datasets/pollution_hourly.csv', parse_dates=[0], index_col=[0],
#                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
#df_mean = pd.read_csv('datasets/pollution_mean.csv', parse_dates=[0], index_col=[0],
#                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
#df_max = pd.read_csv('datasets/pollution_max.csv', parse_dates=[0], index_col=[0],
#                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})


# Close plots
#plt.close('all')
