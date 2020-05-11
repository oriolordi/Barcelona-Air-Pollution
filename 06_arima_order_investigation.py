# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


# Turn off automatic plot showing
plt.ioff()


# Load the datasets (daily average and daily maximum value of pollutants in each station) + weather data
df_mean = pd.read_csv('datasets/pollution_mean.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_max = pd.read_csv('datasets/pollution_max.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
# Make a copy of the dataset just in case
df_mean_vault = df_mean.copy()
df_max_vault = df_max.copy()


# Function to save a dictionary of dictionaries for each pollutant and each station with the dataframe of values
def convert_to_dict(df):
    dict_df = {key: {} for key in df['POLLUTANT'].unique()}
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            df_station_pollutant = df['VALUE'][(df['STATION'] == station) & (df['POLLUTANT'] == pollutant)]
            dict_df[pollutant].update({station: df_station_pollutant})
    return dict_df


# Convert the dataframe to dictionaries
dict_df_mean = convert_to_dict(df_mean)
dict_df_max = convert_to_dict(df_max)


# Function to subplot the results of the time series decomposition
def plot_seasonal(res, axes, station, i, plot=True):
    if plot:
        res.observed.plot(ax=axes[0], legend=False)
        res.trend.plot(ax=axes[1], legend=False)
        res.seasonal.plot(ax=axes[2], legend=False)
        res.resid.plot(ax=axes[3], legend=False)
        if i == 0:
            axes[0].set_ylabel('Observed')
            axes[1].set_ylabel('Trend')
            axes[2].set_ylabel('Seasonal')
            axes[3].set_ylabel('Residual')
    plt.xticks([])
    axes[3].set_xlabel(station)


# Function plot the decompositon of the time series of each pollutant in one plot with stations as subplots
def decompose_plot(dictionary):
    for pollutant, station_df in dictionary.items():
        fig, axes = plt.subplots(ncols=7, nrows=4, sharex=True, figsize=(12, 5))
        for i, (station, df) in enumerate(station_df.items()):
            if len(df) != 0:
                res = sm.tsa.seasonal_decompose(df)
                plot_seasonal(res=res, axes=axes[:, i], station=station, i=i, plot=True)
            else:
                plot_seasonal(res=0, axes=axes[:, i], station=station, i=i, plot=False)
            plt.suptitle('Time series decomposition of Pollutant = {}' .format(pollutant), size=20, fontweight='bold')
            plt.show()


# Decompose and plot the time series for the pollution mean and the max values
decompose_plot(dict_df_mean)
decompose_plot(dict_df_max)


# Function to perform augmented Dickey Fuller (ADF) test to check for stationarity
def adf_test(dictionary):
    dict_adf = {key: pd.DataFrame(index=df_mean['STATION'].unique(), columns=['ADF', 'p-value'])
                for key in df_mean['POLLUTANT'].unique()}
    for pollutant, station_df in dictionary.items():
        for i, (station, df) in enumerate(station_df.items()):
            if len(df) != 0:
                # Pollutant O3 is not stationary. It is differenced to see if one order difference makes it stationary
                if (pollutant == 'O3'):
                    df = df.diff()[1:len(df)]
                result = adfuller(df)
                dict_adf[pollutant].loc[station]['ADF'] = result[0]
                dict_adf[pollutant].loc[station]['p-value'] = result[1]
    return dict_adf


# Decompose and plot the time series for the mean and the max values
dict_adf_mean = adf_test(dict_df_mean)
dict_adf_max = adf_test(dict_df_max)


# From the dataframes saved in dict_adf_mean and dict_adf_max it is seen that only the pollutant O3 is not stationary
# because the p-value is lower than 0.05 everywhere except in th O3 stations (all of the stations are not stationary in O3)
# Thus, the parameter d for the ARIMA model is d=0 everywhere except in the pollutant O3 where it is d=1


# Function to plot the acf and pacf of each pollutant in one plot with stations as subplots
def pacf_acf_plot(dictionary):
    for pollutant, station_df in dictionary.items():
        plt.figure()
        for i, (station, df) in enumerate(station_df.items()):
            # Plot pacf
            plt.subplot(7, 2, 2*i+1)
            if len(df) != 0:
                plot_pacf(df, ax=plt.gca())
            else:
                plt.plot(0)
            plt.ylabel(station, rotation='60')
            plt.yticks([])
            if i != 6:
                plt.xticks([])
            else:
                plt.xlim([-1, 27])
                plt.xticks(range(0, 27, 5))
            if i != 0:
                plt.title('')
            # Plot acf
            plt.subplot(7, 2, 2*i+2)
            if len(df) != 0:
                plot_acf(df, ax=plt.gca())
            else:
                plt.plot(0)
            if i != 0:
                plt.title('')
            plt.yticks([])
            if i != 6:
                plt.xticks([])
            else:
                plt.xlim([-1, 27])
                plt.xticks(range(0, 27, 5))
        plt.suptitle('Pollutant = {}' .format(pollutant), size=20, fontweight='bold')
        plt.show()


# Plot the acf and pacf for the pollution mean and the max values
pacf_acf_plot(dict_df_max)
pacf_acf_plot(dict_df_mean)


# Close all plots
#plt.close('all')


# From the pacf and acf, the following values of p and q in the ARIMA model can be inferred and plugged into the df
# for all pollutants, the AutoRegressive term and the Moving average term are 2 (p=2, q=2)
# However, different options will be tried out, specifically 3 of them:
# (p=2, q=0), (p=0, q=2) and (p=2, q=2)
# As for the seasonal values (P and Q), both are set to 1 due to the peak at lag 7 (seasonality=7) in the pacf and acf
# So, P=1, Q=1 and seasonality=7 for all the pollutants and stations
