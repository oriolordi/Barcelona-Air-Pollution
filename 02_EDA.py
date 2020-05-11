# Import statements
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# Turn off automatic plot showing
plt.ioff()


# Load the dataset
df = pd.read_csv('datasets/pollution.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
# Make a copy of the dataset just in case
df_vault = df.copy()


# Order the dataframe by POLLUTANT and STATION
df.sort_values(by=['POLLUTANT', 'STATION'], inplace=True)


# Save a dictionary of dictionaries for each pollutant and each station with the dataframe of value
dict_df = {key: {} for key in df['POLLUTANT'].unique()}
for pollutant in df['POLLUTANT'].unique():
    for station in df['STATION'].unique():
        df_station_pollutant = df['VALUE'][(df['STATION'] == station) & (df['POLLUTANT'] == pollutant)]
        dict_df[pollutant].update({station: df_station_pollutant})


# Check if any dates are missing
missing_hours_dict = {key: {} for key in dict_df.keys()}
missing_days_dict = {key: {} for key in dict_df.keys()}
missing_hours = []
for pollutant, station_df in dict_df.items():
    for station, df in station_df.items():
        number_of_missing_hours = len(pd.date_range(start='2019-04-01', end='2020-02-29', freq='1H').difference(df.index))
        number_of_missing_days = number_of_missing_hours / 24
        total_hours = len(pd.date_range(start='2019-04-01', end='2020-02-29', freq='1H'))
        missing_hours.append(pd.date_range(start='2019-04-01', end='2020-02-29', freq='1H').difference(df.index))
        if number_of_missing_hours == total_hours:
            number_of_missing_hours = 'ALL'
            number_of_missing_days = 'ALL'
        missing_hours_dict[pollutant].update({station: number_of_missing_hours})
        missing_days_dict[pollutant].update({station: number_of_missing_days})
# Get dataframes that tell the number of missing hours and number of missing days in each pollutant-station combination
df_number_of_missing_hours = pd.DataFrame(missing_hours_dict)
df_number_of_missing_days = pd.DataFrame(missing_days_dict)
# Get the dates that are missing in ALL (the 49) combinations of pollutant-station
df_missing_hours = pd.DataFrame(missing_hours).transpose()
range_of_dates = pd.date_range(start='2019-04-01', end='2020-04-01', freq='1H')
# Get true or false for each column (of the 49 combinations) of whether the date is missing or not
list_is_in_range_of_dates = []
for colname in df_missing_hours:
    list_is_in_range_of_dates.append(range_of_dates.isin(df_missing_hours[colname]))
df_is_in_range_of_dates = pd.DataFrame(list_is_in_range_of_dates).transpose()
# If all the columns are true (the 49 combination) that date is missing everywhere
date_missing_everywhere = df_is_in_range_of_dates.sum(axis=1) == len(df_is_in_range_of_dates.columns)
df_date_missing_everywhere = pd.DataFrame(range_of_dates[date_missing_everywhere])
df_dates_missing_and_not_missing = pd.DataFrame(date_missing_everywhere).set_index(range_of_dates)
df_dates_missing_and_not_missing.columns = ['missing']
df_dates_missing_and_not_missing.reset_index(inplace=True)
# Plot the dates that are missing everywhere (continuous)
# fig, ax = plt.subplots(figsize=(10, 4))
# plt.plot(df_dates_missing_and_not_missing['index'], df_dates_missing_and_not_missing['missing'])
# conversion = {'Missing data': True, 'Available data': False}
# plt.ylim([-1, 2])
# ax.set_yticks(list(conversion.values()))
# ax.set_yticklabels(list(conversion.keys()))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# label_format = {'fontsize': 12, 'fontweight': 'bold'}
# title_format = {'fontsize': 16, 'fontweight': 'bold'}
# ax.set_xlabel('Dates', **label_format)
# ax.set_ylabel('Availability of the data', **label_format)
# plt.title('Observations missing in EVERY station and pollutant', **title_format)
# plt.show()
# Prepare data to plot the dates that are missing everywhere (discontinuous)
pos = np.where(df_dates_missing_and_not_missing['missing'] != df_dates_missing_and_not_missing['missing'].shift(1))[0]
df_dates_missing_and_not_missing['index'][pos] = np.datetime64('NaT')
df_dates_missing_and_not_missing['missing'][pos] = np.nan
df_dates_missing_and_not_missing.drop(0, inplace=True)
# Plot the dates that are missing everywhere (discontinuous)
fig, ax = plt.subplots(figsize=(10, 4))
plt.plot(df_dates_missing_and_not_missing['index'], df_dates_missing_and_not_missing['missing'])
conversion = {'Missing data': True, 'Available data': False}
plt.ylim([-1, 2])
ax.set_yticks(list(conversion.values()))
ax.set_yticklabels(list(conversion.keys()))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
label_format = {'fontsize': 12, 'fontweight': 'bold'}
title_format = {'fontsize': 16, 'fontweight': 'bold'}
ax.set_xlabel('Date', **label_format)
ax.set_ylabel('Availability of the data', **label_format)
plt.title('Observations missing in EVERY station and pollutant', **title_format)
# Add text with the date span every time the date changes from missing to not missing or otherwise
previous_date = df_dates_missing_and_not_missing['index'][1] - timedelta(hours=1)
for i, date in enumerate(df_dates_missing_and_not_missing['index']):
    if pd.isnull(date) or i == len(df_dates_missing_and_not_missing['index'])-1:
        new_date = df_dates_missing_and_not_missing['index'][i]
        date_diff = new_date - previous_date
        text = str(date_diff.days) + str(' days')
        plt.text(previous_date,
                 df_dates_missing_and_not_missing['missing'][i] + 0.05,
                 s=text,
                 rotation=60)
        previous_date = new_date
plt.show()


# Fill missing dates
for pollutant, station_df in dict_df.items():
    for station, df in station_df.items():
        if len(df) > 0:
            # Fill missing dates
            idx = pd.date_range(min(df.index), max(df.index), freq='1H')
            df = df.reindex(idx, fill_value=np.nan)
            # Update values
            dict_df[pollutant].update({station: df})


# Plot the time series of each pollutant in each station
dict_plot = copy.deepcopy(dict_df)
for pollutant, station_df in dict_plot.items():
    plt.figure()
    for i, (station, df) in enumerate(station_df.items()):
        plt.subplot(len(dict_df), 1, i+1)
        if len(df) == 0:
            idx = pd.date_range(min(df_vault.index), max(df_vault.index), freq='1H')
            df = df.reindex(idx, fill_value=-1)
            plt.ylim([0, 1])
            plt.yticks([0, 1])
        plt.plot(df)
        plt.title(station, y=0)
        if i < (len(station_df)-1):
            plt.xticks([])
    plt.suptitle('Hourly time series \n \n Pollutant = {}'.format(pollutant), fontweight='bold')
    plt.show()


# Plot the distribution of each pollutant in each station
for pollutant, station_df in dict_df.items():
    plt.figure()
    for i, (station, df) in enumerate(station_df.items()):
        plt.subplot(len(dict_df), 1, i+1)
        df.hist(bins=100)
        plt.title(station, y=0)
        plt.xticks([])
    plt.suptitle('Distribution of Pollutant = {}' .format(pollutant))
    plt.show()


# Convert the dictionary of dictionaries with filled missing dates back to a dataframe
df_missing_dates_list = []
for pollutant, station_df in dict_df.items():
    for station, df in station_df.items():
        df_to_append = pd.DataFrame(df)
        df_to_append['STATION'] = station
        df_to_append['POLLUTANT'] = pollutant
        df_missing_dates_list.append(df_to_append)
df_missing_dates = pd.concat(df_missing_dates_list)
df_missing_dates['STATION'] = df_missing_dates['STATION'].astype('category')
df_missing_dates['POLLUTANT'] = df_missing_dates['POLLUTANT'].astype('category')


# Get the combinations of stations and pollutants and their percentage of NA's (or NA if the combination has no data)
list_percentage_of_na = []
for pollutant in df_missing_dates['POLLUTANT'].unique():
    for station in df_missing_dates['STATION'].unique():
        df_station_pollutant = df_missing_dates[(df_missing_dates['STATION'] == station) &
                                                (df_missing_dates['POLLUTANT'] == pollutant)]
        if len(df_station_pollutant['VALUE']) != 0:
            list_percentage_of_na.append((pollutant, station,
                                          sum(df_station_pollutant['VALUE'].isna()) /
                                          len(df_station_pollutant['VALUE']) * 100))
        else:
            list_percentage_of_na.append((pollutant, station, np.nan))
# Convert to a dataframe
df_percentage_of_na = pd.DataFrame(list_percentage_of_na, columns=['POLLUTANT', 'STATION', 'Percentage of missing data'])
# Create dataframes indicating whether there is available data in a specific POLLUTANT-STATION pair
# (the rows of df_percentage_of_na where the percentage is NA)
df_data = df_percentage_of_na[np.logical_not(df_percentage_of_na['Percentage of missing data'].isna())]
df_no_data = df_percentage_of_na[df_percentage_of_na['Percentage of missing data'].isna()]


# Plot a graph indicating which POLLUTANT-STATION pairs have available data
g = plt.figure()
plt.scatter(df_data['STATION'], df_data['POLLUTANT'], c='blue', marker='o')
plt.scatter(df_no_data['STATION'], df_no_data['POLLUTANT'], c='red', marker='x')
# Place the legend outside the plot area
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(['Data available', 'No data available'], loc='center left', bbox_to_anchor=(1, 0.5))
# Add title and adjust plot size in figure
g.suptitle("Available data exploration", size=20, fontweight='bold')
g.subplots_adjust(top=0.85, right=0.8)
plt.ylabel('POLLUTANT', fontweight='bold')
plt.xlabel('STATION', fontweight='bold')
#plt.title('Available data exploration')
# Show the plot
plt.show()


# Plot the percentage of missing data
g = sns.relplot(x='STATION', y='POLLUTANT', size='Percentage of missing data', sizes=(20, 200), data=df_percentage_of_na)
g.fig.suptitle("Missing data exploration", size=20, fontweight='bold')
g.fig.subplots_adjust(top=0.85)
plt.ylabel('POLLUTANT', fontweight='bold')
plt.xlabel('STATION', fontweight='bold')
plt.show()


# Just an empty relational plot for test purposes
g = sns.relplot(x='STATION', y='POLLUTANT', size=1, sizes=(50, 50), data=df_percentage_of_na,
                legend=False)
g.fig.suptitle("Relational plot", size=20, fontweight='bold')
g.fig.subplots_adjust(top=0.85)
plt.ylabel('POLLUTANT', fontweight='bold')
plt.xlabel('STATION', fontweight='bold')
plt.show()


# Check if the missing data is clustered together
df_na_chunks = df_missing_dates.copy()
df_na_chunks['Subgroup'] = (df_na_chunks['VALUE'].isna() & df_na_chunks['VALUE'].isna().shift(1)).astype(int)
df_na_chunks['Cumulative'] = df_na_chunks.groupby(df_na_chunks['Subgroup'].eq(0).cumsum()).cumcount()
condition = (df_na_chunks['Cumulative'] != 0) & (df_na_chunks['Cumulative'].shift(-1) == 0)
df_na_chunks['Last cumulative'] = df_na_chunks['Cumulative'].mask(np.logical_not(condition))
# Plot the position and the size of the chunks of missing data
g = sns.lmplot(x='index', y='Last cumulative', hue='STATION', col='POLLUTANT', data=df_na_chunks.reset_index(), fit_reg=False)
g.fig.suptitle("Consecutive hours of missing data by station and pollutant", size=20, fontweight='bold')
g.fig.subplots_adjust(top=0.85, bottom=0.15, left=0.075)
g.set_ylabels('Number of consecutive hours of missing data', fontweight='bold')
g.set_xlabels('')
g.fig.text(x=0.5, y=0, horizontalalignment='center', s='Date', fontweight='bold')
g.set_xticklabels(rotation=45)
plt.show()


# Find the peak of missing data (1600+ consecutive hours of missing data on POLLUTANT PM10 and STATION Les corts)
# it's the peak that can be seen in the prior plot
df_na_chunks.reset_index(inplace=True)
end_point = df_na_chunks[df_na_chunks['Last cumulative'] == df_na_chunks['Last cumulative'].max()].index.values[0]
span = int(df_na_chunks['Last cumulative'].max())
df_large_missing_data = df_na_chunks.iloc[end_point-span:end_point, :]
initial_date = df_large_missing_data['index'].min()
final_date = df_large_missing_data['index'].max()
print('Data missing on PM10 in Les corts starts in {} and ends in {}' .format(initial_date, final_date))


# Reorder dataframe columns
order = ['POLLUTANT', 'STATION', 'VALUE']
df_missing_dates = df_missing_dates[order]


# Save to a csv the data with filled missing dates
df_missing_dates.to_csv('datasets/pollution_missing_dates.csv')
# Line to correctly read this csv (for future use)
#df = pd.read_csv('datasets/pollution_missing_dates.csv', parse_dates=[0], index_col=[0],
#                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})


# Close all plots
#plt.close('all')
