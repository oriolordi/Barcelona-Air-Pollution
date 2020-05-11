# Import statements
import requests
import numpy as np
import pandas as pd


# Read all the csv pollution files
# April 2019
df = pd.read_csv('datasets/2019_04_Abril_qualitat_aire_BCN.csv')
# Add May 2019
df = pd.concat([df, pd.read_csv('datasets/2019_05_Maig_qualitat_aire_BCN.csv')], sort=False)
# Add June 2019
df = pd.concat([df, pd.read_csv('datasets/2019_06_Juny_qualitat_aire_BCN.csv', sep=';')], sort=False)
# Add July 2019
df = pd.concat([df, pd.read_csv('datasets/2019_07_Juliol_qualitat_aire_BCN.csv', sep=';')], sort=False)
# Add August 2019
df = pd.concat([df, pd.read_csv('datasets/2019_08_Agost_qualitat_aire_BCN.csv')], sort=False)
# Add September 2019
df = pd.concat([df, pd.read_csv('datasets/2019_09_Setembre_qualitat_aire_BCN.csv')], sort=False)
# Add October 2019
df = pd.concat([df, pd.read_csv('datasets/2019_10_Octubre_qualitat_aire_BCN.csv')], sort=False)
# Add November 2019
df = pd.concat([df, pd.read_csv('datasets/2019_11_Novembre_qualitat_aire_BCN.csv')], sort=False)
# Add December 2019
df = pd.concat([df, pd.read_csv('datasets/2019_12_Desembre_qualitat_aire_BCN.csv')], sort=False)
# Add January 2020
df = pd.concat([df, pd.read_csv('datasets/2020_01_Gener_qualitat_aire_BCN.csv')], sort=False)
# Add Februrary 2020
df = pd.concat([df, pd.read_csv('datasets/2020_02_Febrer_qualitat_aire_BCN.csv')], sort=False)
# Add March 2020
df = pd.concat([df, pd.read_csv('datasets/2020_03_Marc_qualitat_aire_BCN.csv')], sort=False)


# Change the code of the pollutant (CODI_CONTAMINANT) and the code of the station (ESTACIO) from numbers to names
# Get the file containing the number-name equivalencies for pollutants
df_pollutants = pd.read_csv('datasets/Qualitat_Aire_Contaminants.csv')
# Create a dictionary containing the relationship between code and name and use it to replace values (pollutants)
pollutants_code_name = dict(zip(df_pollutants['Codi_Contaminant'], df_pollutants['Desc_Contaminant']))
df['CODI_CONTAMINANT'].replace(pollutants_code_name, inplace=True)
# Get the file containing the number-name equivalencies for stations
df_stations = pd.read_csv('datasets/Qualitat_Aire_Estacions.csv')
# Create a dictionary containing the relationship between code and name and use it to replace values (station)
stations_code_name = dict(zip(df_stations['Estacio'], df_stations['Nom_districte']))
df['ESTACIO'].replace(stations_code_name, inplace=True)
# Check that all is correct
print(df.groupby('CODI_CONTAMINANT').count())
print(df.groupby('ESTACIO').count())
# There are two categories ('2' in pollutants and '58' in stations) that haven't found a name.
# However, the total samples of each of them is significantly lower than the rest.
# Thus, they are removed from the dataframe
index_pollutant_2 = df[df['CODI_CONTAMINANT'] == '2'].index
df.drop(index_pollutant_2, inplace=True)
index_station_58 = df[df['ESTACIO'] == '58'].index
df.drop(index_station_58, inplace=True)


# Make sure that the validation columns can be dropped
hourly_value_columns = [col for col in df.columns if col.startswith('H')]
validation_value_columns = [col for col in df.columns if col.startswith('V')]
hour_validation = list(zip(hourly_value_columns, validation_value_columns))
# Check whether there are only 'N' and 'V' values in all the validation columns
print('Validation V-N check:')
for v in validation_value_columns:
    print(set(df[v]))
# Check that every 'N' value on a validation column corresponds to a NA on the hour column
print('Validation N check:')
for h, v in hour_validation:
    df_N = df[h][df[v] == 'N']
    print(np.logical_not(df_N.isna()).sum())
# Check that every 'V' value on a validation column corresponds to a value not NA on the hour column
print('Validation V check:')
for h, v in hour_validation:
    df_V = df[h][df[v] == 'V']
    print(df_V.isna().sum())
# The validation columns, thus, can be dropped
df.drop(validation_value_columns, axis=1, inplace=True)


# Drop the columns that are useless (the ones that inform that the data is from barcelona)
df.drop(['CODI_PROVINCIA', 'PROVINCIA', 'CODI_MUNICIPI', 'MUNICIPI'], axis=1, inplace=True)


# Convert to a datetime format
df = pd.melt(df, id_vars=['ESTACIO', 'CODI_CONTAMINANT', 'ANY', 'MES', 'DIA'], value_vars=hourly_value_columns,
               var_name='HORA', value_name='VALUE')
df['HORA'] = df['HORA'].str.lstrip('H')
df['DATETIME'] = pd.to_datetime(dict(year=df['ANY'], month=df['MES'], day=df['DIA'], hour=df['HORA']))
df.sort_values(by='DATETIME', ascending=True, inplace=True)
df.drop(['ANY', 'MES', 'DIA', 'HORA'], axis=1, inplace=True)


# Rearrange columns to display datetime on the first column
column_names = list(df.columns)
print(column_names)
order = [3, 0, 1, 2]
column_names = [column_names[i] for i in order]
print(column_names)
df = df[column_names]


# Change the column names to English
df.rename({'ESTACIO': 'STATION', 'CODI_CONTAMINANT': 'POLLUTANT'}, axis=1, inplace=True)


# Write the dataframe to a .csv file
df.to_csv('datasets/pollution.csv', index=False)
# Line to correctly read this csv (for future use)
#df = pd.read_csv('datasets/pollution.csv', parse_dates=[0], index_col=[0],
#                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
