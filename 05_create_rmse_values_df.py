# Import statements
import pandas as pd
import matplotlib.pyplot as plt


# Turn off automatic plot showing
plt.ioff()


# Load the dataset
df = pd.read_csv('datasets/pollution.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
# Make a copy of the dataset just in case
df_vault = df.copy()


# Models to be used and have their RMSE values stored
models = ['Just_Naive',
          'Add_Holt-Winters', 'Add_Arima AR 2', 'Add_Arima MA 2', 'Add_Arima AR 2 MA 2', 'Add_Prophet',
          'Mul_Holt-Winters', 'Mul_Arima AR 2', 'Mul_Arima MA 2', 'Mul_Arima AR 2 MA 2', 'Mul_Prophet',
          'Supervised DT', 'Supervised KNN', 'Supervised RF',
          'Super7 DT', 'Super7 KNN', 'Super7 RF',
          'DeepLearning MLP', 'DeepLearning CNN', 'DeepLearning LSTM']
          #, 'DeepLearning CNN-LSTM', 'DeepLearning ConvLSTM']


# Save a dictionary of dictionaries for each pollutant and each station with a dataframe containing all models used
# and a RMSE value (which for now is empty (nan))
dict_df = {key: {} for key in df['POLLUTANT'].unique()}
for pollutant in df['POLLUTANT'].unique():
    for station in df['STATION'].unique():
        df_dict = pd.DataFrame(index=models, columns=['RMSE'])
        dict_df[pollutant].update({station: df_dict})


# Convert the dictionary to a dataframe that can be save in a csv
list_df_to_save = []
for pollutant, station_df in dict_df.items():
    for station, df in station_df.items():
        df_to_append = pd.DataFrame(df)
        df_to_append['STATION'] = station
        df_to_append['POLLUTANT'] = pollutant
        list_df_to_save.append(df_to_append)
df_to_save = pd.concat(list_df_to_save)


# Save the csv file
df_to_save.to_csv('datasets/rmse.csv')
# Line to correctly read this csv (for future use)
#df = pd.read_csv('datasets/rmse.csv', index_col=[0])
