# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Turn off automatic plot showing
plt.ioff()


# Load the datasets (daily average and daily maximum value of pollutants in each station) + weather data + RMSE values
df_mean = pd.read_csv('datasets/pollution_mean.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_max = pd.read_csv('datasets/pollution_max.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_weather = pd.read_csv('datasets/weather_data.csv', parse_dates=[0], index_col=[0])
# Make a copy of the dataset just in case
df_mean_vault = df_mean.copy()
df_max_vault = df_max.copy()
df_weather_vault = df_weather.copy()


# Function to save a dictionary of dictionaries for each pollutant and each station with the dataframe of values
# while also merging the weather information.
# also, save Februrary of 2020 as validation data for later use
def convert_to_dict(df):
    dict_df = {key: {} for key in df['POLLUTANT'].unique()}
    dict_df_validation = {key: {} for key in df['POLLUTANT'].unique()}
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            df_station_pollutant = df['VALUE'][(df['STATION'] == station) & (df['POLLUTANT'] == pollutant)]
            df_merged = pd.merge(df_station_pollutant, df_weather, how='inner', left_index=True, right_index=True)
            try:
                df_merged_modelling = df_merged[
                    np.logical_not((df_merged.index.month == 2) & (df_merged.index.year == 2020))]
                df_merged_validation = df_merged[
                    (df_merged.index.month == 2) & (df_merged.index.year == 2020)]
            except:
                df_merged_modelling = df_merged
                df_merged_validation = df_merged
            dict_df[pollutant].update({station: df_merged_modelling})
            dict_df_validation[pollutant].update({station: df_merged_validation})
    return dict_df, dict_df_validation


# Convert the dataframe to dictionaries
dict_df_mean, dict_df_mean_validation = convert_to_dict(df_mean)
dict_df_max, dict_df_max_validation = convert_to_dict(df_max)


# Function to study the relation between weather data and pollution data
def pollution_weather_study(dictionary_mean, dictionary_max):
    df_list = list()
    for pollutant, station_df in dictionary_mean.items():
        for station, df in station_df.items():
            # Study the relation between weather and polution only for combinations of STATION and POLLUTANT that have data
            if len(df > 0):
                corr = df.corr()
                corr = pd.DataFrame(corr['VALUE'])
                corr.loc[:, 'POLLUTANT'] = pollutant
                corr.loc[:, 'STATION'] = station
                corr.loc[:, 'TYPE'] = 'Mean'
                df_list.append(corr)
    for pollutant, station_df in dictionary_max.items():
        for station, df in station_df.items():
            # Study the relation between weather and polution only for combinations of STATION and POLLUTANT that have data
            if len(df > 0):
                corr = df.corr()
                corr = pd.DataFrame(corr['VALUE'])
                corr.loc[:, 'POLLUTANT'] = pollutant
                corr.loc[:, 'STATION'] = station
                corr.loc[:, 'TYPE'] = 'Max'
                df_list.append(corr)
    df_corr = pd.concat(df_list, sort=False)
    return df_corr


# Model and evaluate the data using the mean and the max values of the pollutants
# the goal of this function is basically to fill  values of the dictionaries (dict_rmse_mean and dict_rmse_max)
# these dictionaries contain for each POLLUTANT and each STATION a dataframe with a column of RMSE values for each model
df_correlation = pollution_weather_study(dict_df_mean, dict_df_max)


# Order and sort the dataframe
order = ['POLLUTANT', 'STATION', 'TYPE', 'VALUE']
df_correlation = df_correlation[order]
df_correlation.columns = [*df_correlation.columns[:-1], 'Correlation with pollutant']
df_correlation = df_correlation[df_correlation.index != 'VALUE']
df_correlation_positive = df_correlation.sort_values(by='Correlation with pollutant', ascending=False)
df_correlation_negative = df_correlation.sort_values(by='Correlation with pollutant')


# Average correlation of each POLLUTANT across all STATIONS with each weather attribute
df_correlation_pollutants = df_correlation_positive.copy()
df_correlation_pollutants.index.name = 'Weather attribute'
df_correlation_pollutants = df_correlation_pollutants.reset_index().groupby(['POLLUTANT', 'Weather attribute'],
                                                                            as_index=False).mean()
# change names of the weather attributes for a better legend
weather_replace = {'average_temperature': 'Temperature', 'rain': 'Rainfall', 'wind_direction': 'Wind direction',
                   'wind_speed': 'Wind speed'}
df_correlation_pollutants['Weather attribute'].replace(weather_replace, inplace=True)


# Plot the average correlation of each POLLUTANT across all STATIONS with each weather attribute
# set width of bar
barWidth = 0.2
# set bars
weather_attributes = df_correlation_pollutants['Weather attribute'].unique()
bars1 = df_correlation_pollutants['Correlation with pollutant'][df_correlation_pollutants['Weather attribute'] ==
                                                                weather_attributes[0]]
bars2 = df_correlation_pollutants['Correlation with pollutant'][df_correlation_pollutants['Weather attribute'] ==
                                                                weather_attributes[1]]
bars3 = df_correlation_pollutants['Correlation with pollutant'][df_correlation_pollutants['Weather attribute'] ==
                                                                weather_attributes[3]]
#bars4 = df_correlation_pollutants['Correlation with pollutant'][df_correlation_pollutants['Weather attribute'] ==
#                                                                weather_attributes[2]]
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
#r4 = [x + barWidth for x in r3]
# Make the plot
plt.bar(r1, bars1, color='red', width=barWidth, edgecolor='white', label=weather_attributes[0])
plt.bar(r2, bars2, color='blue', width=barWidth, edgecolor='white', label=weather_attributes[1])
plt.bar(r3, bars3, color='green', width=barWidth, edgecolor='white', label=weather_attributes[3])
#plt.bar(r4, bars4, color='green', width=barWidth, edgecolor='white', label=weather_attributes[2])
# Add xticks on the middle of the group bars
plt.xlabel('POLLUTANT', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], df_correlation_pollutants['POLLUTANT'].unique())
# Add ylabel and limits
plt.ylabel('CORRELATION', fontweight='bold')
plt.ylim([-1, 1])
plt.title('Correlation of weather attributes with the different pollutants', size=20, fontweight='bold')
# Create legend & Show graphic
plt.legend()
plt.show()
