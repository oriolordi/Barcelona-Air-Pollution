# Import statements
import json
import boto3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Function to preprocess the data
def resample_order(df, pollutant, station):
    # Resample to daily frequency
    df_mean = df.resample('D').mean()
    df_mean.loc[:, 'TYPE'] = 'Mean'
    df_max = df.resample('D').max()
    df_max.loc[:, 'TYPE'] = 'Max'
    df_total = pd.concat([df_mean, df_max], sort=True)
    df_total.loc[:, 'POLLUTANT'] = pollutant
    df_total.loc[:, 'STATION'] = station
    df_total.loc[:, 'DATA'] = 'Gathered'
    return df_total


# The follwing of the script is a function to preprocess the data of the last few days that only executes if the data was read
def data_preprocess(df, df_s, df_p):
    # Change the code of the pollutant (CODI_CONTAMINANT) and the code of the station (ESTACIO) from numbers to names
    # Create a dictionary containing the relationship between code and name and use it to replace values (pollutants)
    pollutants_code_name = dict(zip(df_p['Codi_Contaminant'], df_p['Desc_Contaminant']))
    df['CODI_CONTAMINANT'].replace(pollutants_code_name, inplace=True)
    # Create a dictionary containing the relationship between code and name and use it to replace values (station)
    stations_code_name = dict(zip(df_s['Estacio'], df_s['Nom_districte']))
    df['ESTACIO'].replace(stations_code_name, inplace=True)
    # Removed from the dataframe the POLLUTANT (2) and STATION (58) that don't have a name
    index_pollutant_2 = df[df['CODI_CONTAMINANT'] == 2].index
    df.drop(index_pollutant_2, inplace=True)
    index_station_58 = df[df['ESTACIO'] == 58].index
    df.drop(index_station_58, inplace=True)


    # Changing the hourly values to numeric
    hourly_value_columns = [col for col in df.columns if col.startswith('H')]
    df[hourly_value_columns] = df[hourly_value_columns].apply(pd.to_numeric, errors='coerce')


    # Drop the validation columns
    validation_value_columns = [col for col in df.columns if col.startswith('V')]
    df.drop(validation_value_columns, axis=1, inplace=True)


    # Drop the columns that are useless (the ones that inform that the data is from barcelona)
    df.drop(['CODI_PROVINCIA', 'PROVINCIA', 'CODI_MUNICIPI', 'MUNICIPI'], axis=1, inplace=True)


    # Convert to a datetime format
    df = pd.melt(df, id_vars=['ESTACIO', 'CODI_CONTAMINANT', 'ANY', 'MES', 'DIA'], value_vars=hourly_value_columns,
                   var_name='HORA', value_name='VALUE')
    df.loc[:, 'HORA'] = df['HORA'].str.lstrip('H')
    df.loc[:, 'DATETIME'] = pd.to_datetime(dict(year=df['ANY'], month=df['MES'], day=df['DIA'], hour=df['HORA']))
    df.sort_values(by='DATETIME', ascending=True, inplace=True)
    df.drop(['ANY', 'MES', 'DIA', 'HORA'], axis=1, inplace=True)


    # Rearrange columns to display datetime on the first column
    column_names = list(df.columns)
    order = [3, 0, 1, 2]
    column_names = [column_names[i] for i in order]
    df = df[column_names]


    # Change the column names to English
    df.rename({'ESTACIO': 'STATION', 'CODI_CONTAMINANT': 'POLLUTANT'}, axis=1, inplace=True)


    # Subset the dataframe to the data from the day before only
    # Cheating a bit:
    # since the data is displayed from 1 hour to 24 hours for each day, but python interprets midnight as the next day,
    # all hours will be delayed one hour so that when the samples are resampled to a daily frequency,
    # the 24 hours taken for the resampling are the ones corresponding to that day.
    # OBSERVATION: this is only valid because of the daily resampling, if the data was to be analyzed hourly,
    # it would make no sense to delay observations by one hour (or it would, but then it would have to be taken
    # into account when predicting future values)
    df.loc[:, 'DATETIME'] = df['DATETIME'] - timedelta(hours=1)
    date_yesterday = pd.Timestamp(datetime.now(pytz.timezone('Europe/Madrid')).date()) - timedelta(days=1)
    df = df[df['DATETIME'].dt.date == date_yesterday]


    # Interpolate missing values and resample to a daily frequency
    df_list = list()
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            df_p_s = df[(df['POLLUTANT'] == pollutant) & (df['STATION'] == station)]
            if len(df_p_s) > 0:
                # Set DATETIME column as index
                df_p_s.set_index('DATETIME', inplace=True)
                # Interpolate missing values
                df_p_s.loc[:,'VALUE'] = df_p_s['VALUE'].interpolate(method='linear')
                # Resample to daily mean and max and add to a list
                df_to_add = resample_order(df_p_s, pollutant, station)
                df_list.append(df_to_add)


    # Concatenate all the created dataframes in the list
    df_new_data = pd.concat(df_list, sort=True)


    # Return the processed dataframe
    return df_new_data


# Function to fill the data of the last day with the same values as two days ago
def get_last_day(df):
    two_days_ago = pd.Timestamp(datetime.now(pytz.timezone('Europe/Madrid')).date()) - timedelta(days=2)
    df_list = list()
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            for type in df['TYPE'].unique():
                # Get a dataframe for each combination of POLLUTANT, STATION and TYPE
                df_p_s_t = df[(df['POLLUTANT'] == pollutant) & (df['STATION'] == station) & (df['TYPE'] == type)]
                # Reduce that dataframe to containing only the values of the last day
                df_to_add = df_p_s_t[df_p_s_t.index == two_days_ago]
                # Set the index as the next day
                df_to_add.index += timedelta(days=1)
                # Set data to Gathered
                df_to_add.loc[:, 'DATA'] = 'Gathered'
                df_list.append(df_to_add)
    # Concatenate all the created dataframes in the list
    df_new_data = pd.concat(df_list, sort=True)
    # Return the complete dataframe
    return df_new_data


# Function to add a new day of data to predict in each combination of POLLUTANT, STATION and TYPE
def add_new_day(df):
    df_list = list()
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            for type in df['TYPE'].unique():
                # Get a dataframe for each combination of POLLUTANT, STATION and TYPE
                df_p_s_t = df[(df['POLLUTANT'] == pollutant) & (df['STATION'] == station) & (df['TYPE'] == type)]
                if len(df_p_s_t) > 0:
                    last_day = df_p_s_t.iloc[[-1]].index
                    #last_day = pd.DatetimeIndex([df_p_s_t.index.max()])#equivalent to the line above, in theory since it's ordered
                    next_day = last_day + timedelta(days=1)
                    data = {'POLLUTANT': pollutant, 'STATION': station, 'TYPE': type, 'DATA': 'Predicted', 'VALUE': np.nan}
                    df_to_add = df_p_s_t.append(pd.DataFrame(data, index=next_day))
                    df_list.append(df_to_add)
    # Concatenate all the created dataframes in the list
    df_new_data = pd.concat(df_list, sort=False)
    # Return the complete dataframe
    return df_new_data


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


# Function to make predictions for the next days
def make_predictions(df):
    for pollutant in df['POLLUTANT'].unique():
        for station in df['STATION'].unique():
            for meanmax in df['TYPE'].unique():
                data = df['VALUE'][(df['POLLUTANT'] == pollutant) &
                                   (df['STATION'] == station) &
                                   (df['TYPE'] == meanmax) &
                                   (df['DATA'] == 'Gathered')]
                if len(data) > 0:
                    print(pollutant)
                    print(station)
                    print(meanmax)
                    predictions = sarima_prediction(data, pollutant, p=2, q=0, length=7)
                    df['VALUE'][(df['POLLUTANT'] == pollutant) &
                                (df['STATION'] == station) &
                                (df['TYPE'] == meanmax) &
                                (df['DATA'] == 'Predicted')] = predictions
    return df


# Read necessary data from S3
def read_pollution():
    # Access the s3 service
    s3 = boto3.resource('s3')


    # Read pollution dataframe (df)
    # Set bucket name and file name to read and write from
    bucket_name = "bucketairpollution"
    file_name = "pollution_ultimate.csv"
    # Select the wanted file from the wanted bucket
    file = s3.Object(bucket_name, file_name)
    # Read the csv file and store it in a dataframe
    df = pd.read_csv(file.get()['Body'], parse_dates=[0], index_col=[0],
                     dtype={'STATION': 'category', 'POLLUTANT': 'category'})


    # Read stations dataframe (df_s)
    # Set bucket name and file name to read and write from
    bucket_name = "bucketairpollutionstationspollutants"
    file_name = "Qualitat_Aire_Estacions.csv"
    # Select the wanted file from the wanted bucket
    file = s3.Object(bucket_name, file_name)
    # Read the csv file and store it in a dataframe
    df_s = pd.read_csv(file.get()['Body'])


    # Read pollutants dataframe (df_p)
    # Set bucket name and file name to read and write from
    bucket_name = "bucketairpollutionstationspollutants"
    file_name = "Qualitat_Aire_Contaminants.csv"
    # Select the wanted file from the wanted bucket
    file = s3.Object(bucket_name, file_name)
    # Read the csv file and store it in a dataframe
    df_p = pd.read_csv(file.get()['Body'])

    # Return values
    return df, df_s, df_p


# Write necessary data to S3
def write_pollution(df):
    # Access the s3 service
    s3 = boto3.resource('s3')
    # Set bucket name and file name to read and write from
    bucket_name = "bucketairpollution"
    file_name = "pollution_ultimate.csv"
    # Select the wanted file from the wanted bucket
    file = s3.Object(bucket_name, file_name)
    # Upload the csv file to S3
    file.put(Body=df.to_csv())


# Execute the lambda function
def lambda_handler(event, context):
    # Read necessary files
    df_ultimate, df_stations, df_pollutants = read_pollution()
    #df_ultimate = pd.read_csv('datasets/pollution_ultimate.csv', parse_dates=[0], index_col=[0],
    #                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
    #df_stations = pd.read_csv('datasets/Qualitat_Aire_Estacions.csv')
    #df_pollutants = pd.read_csv('datasets/Qualitat_Aire_Contaminants.csv')


    # Read the csv of the last few days data
    # if the csv can't be read, the df_mean and df_max datasets are filled with the same values as the previous day
    try:
        df_last_few_days = pd.read_csv('https://opendata-ajuntament.barcelona.cat/data/dataset/0582a266-ea06-4cc5-a219-'
                                       '913b22484e40/resource/c2032e7c-10ee-4c69-84d3-9e8caf9ca97a/download')
        success = True
    except:
        success = False


    # Preprocess the data, make predictions and update the csv file
    if success:
        df_new = data_preprocess(df_last_few_days, df_stations, df_pollutants)
    else:
        df_new = get_last_day(df_ultimate, df_stations, df_pollutants)


    # Order the columns so that they fit with the existing data
    df_new = df_new[df_ultimate.columns]


    # Erase last day from the ultimate dataframe (which now contains predicted data) and replace it with the current data
    date_last_day = pd.Timestamp(datetime.now(pytz.timezone('Europe/Madrid')).date()) - timedelta(days=1)
    df_ultimate.drop(date_last_day, inplace=True)


    # Add last day's data
    df_ultimate = pd.concat([df_ultimate, df_new])
    df_ultimate.sort_index(inplace=True)
    df_ultimate.sort_values(by=['POLLUTANT', 'STATION', 'TYPE', 'DATA'], inplace=True)


    # Add a new day to predict
    df_ultimate = add_new_day(df_ultimate)


    # Make predictions for the next days
    df_ultimate = make_predictions(df_ultimate)


    # Upload the csv file to S3
    write_pollution(df_ultimate)
    #df_ultimate.to_csv('datasets/pollution_ultimate_test.csv')


# Run the lambda_function (offline only)
e = 1
c = 2
lambda_handler(e, c)
