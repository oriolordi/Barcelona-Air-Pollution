# Import statements
import pandas as pd
from datetime import timedelta


# Read necessary files
df_mean = pd.read_csv('datasets/pollution_mean.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})
df_max = pd.read_csv('datasets/pollution_max.csv', parse_dates=[0], index_col=[0],
                 dtype={'STATION': 'category', 'POLLUTANT': 'category'})


# Add a column to indicate whether the data is the mean value or the max value
df_mean['TYPE'] = 'Mean'
df_max['TYPE'] = 'Max'


# Join the dataframes
df = pd.concat([df_mean, df_max])

# Sort the dataframes
df.sort_values(by=['POLLUTANT', 'STATION', 'TYPE'], inplace=True)

# Add column to know wether the data is gathered or predicted
df['DATA'] = 'Gathered'


# Add 7 days of empty data (np.nan) and tag them as 'Predicted' in the Data column
# (this values will be filled when the prediction is done)
number_of_days_to_predict = 7
for pollutant in df['POLLUTANT'].unique():
    for station in df['STATION'].unique():
        for type in df['TYPE'].unique():
            df_p_s_t = df[(df['POLLUTANT'] == pollutant) & (df['STATION'] == station) & (df['TYPE'] == type)]
            if len(df_p_s_t):
                last_date = df_p_s_t.iloc[[-1]].index
                #last_date = pd.DatetimeIndex([df_p_s_t.index.max()])#equivalent to the line above, in theory since its's ordered
                for i in range(1, number_of_days_to_predict+1):
                    new_date = last_date + timedelta(days=i)
                    data = {'POLLUTANT': pollutant, 'STATION': station, 'TYPE': type, 'DATA': 'Predicted'}
                    df = df.append(pd.DataFrame(data, index=new_date))


# Order and sort the dataframe
order = ['POLLUTANT', 'STATION', 'TYPE', 'DATA', 'VALUE']
df = df[order]
df.sort_values(by=['POLLUTANT', 'STATION', 'TYPE'], inplace=True)


# Save the csv file
df.to_csv('datasets/pollution_ultimate.csv')
