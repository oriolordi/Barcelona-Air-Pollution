# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Turn off automatic plot showing
plt.ioff()


# Load the datasets (daily average and daily maximum value of pollutants in each station) + weather data + RMSE values
df_rmse_mean = pd.read_csv('datasets/pollution_rmse_mean.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_rmse_max = pd.read_csv('datasets/pollution_rmse_max.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
# Make a copy of the dataset just in case
df_rmse_mean_vault = df_rmse_mean.copy()
df_rmse_max_vault = df_rmse_max.copy()


# Get the model names as a column (it's the index at this point, so reset index and rename the column)
df_rmse_mean = df_rmse_mean.reset_index().rename({'index': 'MODEL'}, axis=1)
df_rmse_max = df_rmse_max.reset_index().rename({'index': 'MODEL'}, axis=1)


# Change the data type of the model names column to categorical
#df_rmse_mean['MODEL'] = df_rmse_mean['MODEL'].astype('category')
#df_rmse_max['MODEL'] = df_rmse_max['MODEL'].astype('category')


# Get the best score (lowest RMSE) and  model name for each combination of POLLUTANT and STATION (for each time series)
df_best_models_mean = df_rmse_mean.loc[
    df_rmse_mean.groupby(['POLLUTANT', 'STATION'])['RMSE'].idxmin().dropna()].reset_index(drop=True)
df_best_models_max = df_rmse_max.loc[
    df_rmse_max.groupby(['POLLUTANT', 'STATION'])['RMSE'].idxmin().dropna()].reset_index(drop=True)


# Order the dataframe by POLLUTANT and STATION
df_best_models_mean = df_best_models_mean.sort_values(by=['POLLUTANT', 'STATION'])
df_best_models_max = df_best_models_max.sort_values(by=['POLLUTANT', 'STATION'])


# Function to plot the mean RMSE value across all time series of each MODEL
def plot_rmse_model(df, meanmax):
    # Get normalized mean values of the RMSE for each model across each combination of POLLUTANT and STATION
    df_model_rmse = df.pivot_table(values='RMSE', index='MODEL', columns=['POLLUTANT', 'STATION'])
    scaler = MinMaxScaler()
    df_model_rmse_normalized = pd.DataFrame(scaler.fit_transform(df_model_rmse), index=df_model_rmse.index,
                                            columns=df_model_rmse.columns)
    df_model_rmse_mean = df_model_rmse_normalized.mean(axis=1)
    # Plot the normalized mean values of RMSE per each model, separating the models by group with colours
    g = plt.figure()
    g.subplots_adjust(bottom=0.3)
    df_time_series = df_model_rmse_mean[(df_model_rmse_mean.index.str.startswith('Add')) |
                                        (df_model_rmse_mean.index.str.startswith('Mul'))]
    df_supervised = df_model_rmse_mean[(df_model_rmse_mean.index.str.startswith('Super'))]
    df_deep_learning = df_model_rmse_mean[(df_model_rmse_mean.index.str.startswith('Deep'))]
    plt.bar(np.arange(len(df_time_series)), df_time_series, color='blue', edgecolor='white', label='Time Series')
    plt.bar(np.arange(len(df_time_series), len(df_time_series) + len(df_supervised)), df_supervised,
            color='green', edgecolor='white', label='Supervised')
    plt.bar(np.arange(len(df_time_series) + len(df_supervised), len(df_time_series) + len(df_supervised) +
                      len(df_deep_learning)), df_deep_learning,
            color='yellow', edgecolor='white', label='Neural Networks')
    plt.ylim([0, 1])
    plt.ylabel('NORMALIZED RMSE', fontweight='bold')
    plt.xlabel('MODELS', fontweight='bold')
    df_xticks = pd.concat([df_time_series, df_supervised, df_deep_learning])
    plt.xticks(np.arange(len(df_xticks)), df_xticks.index, rotation=75)
    plt.title('Average normalized RMSE for each model (daily ' + meanmax + ' value)', size=20, fontweight='bold')
    plt.legend()
    plt.show()


# Plot the mean RMSE value across all time series of each MODEL
plot_rmse_model(df_rmse_mean, 'average')
plot_rmse_model(df_rmse_max, 'maximum')


# Function to plot the best models for each combination of POLLUTANT and STATION (for each time series)
def plot_best_models(df, meanmax):
    g = sns.relplot(x='STATION', y='POLLUTANT', size='RMSE', sizes=(0, 0), data=df, legend=False)
    g.fig.suptitle("Best model for each time series (daily " + meanmax + " values)", size=20, fontweight='bold')
    for i, station in enumerate(df['STATION'].unique()):
        for j, pollutant in enumerate(df['POLLUTANT'].unique()):
            try:
                name = df[(df['POLLUTANT'] == pollutant) & (df['STATION'] == station)].iloc[0, 0]
                if name.startswith('Just_Naive'):
                    name = 'Naive'
                elif name.startswith('Add_Arima'):
                    name = 'Arima (add)'
                elif name.startswith('Mul_Arima'):
                    name = 'Arima (mul)'
                elif name.startswith('Add_Holt'):
                    name = 'HW (add)'
                elif name.startswith('Mul_Holt'):
                    name = 'HW (mul)'
                elif name.startswith('Add_Prophet'):
                    name = 'Prophet (add)'
                elif name.startswith('Mul_Prophet'):
                    name = 'Prophet (mul)'
                elif name.startswith('Supervised'):
                    name = name.split(' ')[1] + ' (1 day)'
                elif name.startswith('Super7'):
                    name = name.split(' ')[1] + ' (7 days)'
                elif name.startswith('DeepLearning'):
                    name = name.split(' ')[1]
                df.loc[(df['POLLUTANT'] == pollutant) & (df['STATION'] == station), 'MODEL'] = name
                plt.text(i, j, name, ha='center', size=10)
            except:
                _
    plt.ylabel('POLLUTANT', fontweight='bold')
    plt.xlabel('STATION', fontweight='bold')
    g.fig.subplots_adjust(top=0.85)
    plt.show()
    return df


# Plot the best models for each combination of POLLUTANT and STATION (for each time series)
df_best_models_mean = plot_best_models(df_best_models_mean, 'average')
df_best_models_max = plot_best_models(df_best_models_max, 'maximum')


# Save the csv files of the best models
df_best_models_mean.to_csv('datasets/best_models_mean.csv')
df_best_models_max.to_csv('datasets/best_models_max.csv')
# Line to correctly read the csvs (for future use)
#df_best_models_mean = pd.read_csv('datasets/best_models_mean.csv', index_col=[0])
#df_best_models_max = pd.read_csv('datasets/best_models_max.csv', index_col=[0])
