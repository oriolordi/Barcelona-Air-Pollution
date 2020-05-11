# Import statements
import requests
import pandas as pd


# Set up the url for the historic data as well as the API key for AEMET data
url_historic = 'https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/' \
      '2019-04-02T00%3A00%3A00UTC/fechafin/2020-03-31T00%3A00%3A00UTC/estacion/0201D'
api_key = 'api_key=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvcmlvbC5vcmRpQGdtYWlsLmNvbSIsImp0aSI6IjlkMzMyZDcwLTA2MzctNDFkNi05MD' \
          'k5LWQ5MmFhZDA1YjZiYyIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNTg0MDA4Njk3LCJ1c2VySWQiOiI5ZDMzMmQ3MC0wNjM3LTQx' \
          'ZDYtOTA5OS1kOTJhYWQwNWI2YmMiLCJyb2xlIjoiIn0.NzUfrMQv02k9Ju_7NZoMn2LA-uoyOHusMlAJEDzo95k'
cache_control = 'cache-control=no-cache'
complete_url_historic = url_historic + '?' + api_key + '&' + cache_control


# Get the JSON files from the AEMET API (historic)
json_access_historic = requests.get(complete_url_historic).json()
if json_access_historic['estado'] == 200:
    json_historic = requests.get(json_access_historic['datos']).json()
else:
    print('ERROR')
# Convert the JSON files to a dataframe (historic)
df_list = []
for i, element in enumerate(json_historic):
    df_list.append(pd.DataFrame(element, index=[i]))
df = pd.concat(df_list, sort=True)
df.sort_values(by='fecha', inplace=True)


# Drop useless columns and reorder and rename useful columns
useless_columns = ['altitud', 'horaracha', 'horatmax', 'horatmin', 'indicativo', 'nombre', 'provincia',
                   'racha', 'tmax', 'tmin']
df.drop(useless_columns, axis=1, inplace=True)
df = df[['fecha', 'tmed', 'prec', 'velmedia', 'dir']]
new_column_names = ['datetime', 'average_temperature', 'rain', 'wind_speed', 'wind_direction']
df.columns = new_column_names


# Change data types and make the datetime the index
# Change the datetime column to a datetime format
df['datetime'] = pd.to_datetime(df['datetime'])
# Set the datetime column as the index of the dataframe
df.set_index('datetime', inplace=True)
# Replace commas for dots in the remaining columns (pandas doesn't understand commas as decimal separators)
df= df.apply(lambda x: x.str.replace(',', '.'))
# Change the data types of the remaining columns to numeric (errors='coerce' forces np.nan when non numeric values is found)
df = df.apply(pd.to_numeric, errors='coerce')


# Check if any dates are missing
number_of_missing_dates = len(pd.date_range(start='2019-04-02', end='2020-03-31').difference(df.index))
missing_dates = pd.date_range(start='2019-04-02', end='2020-03-31').difference(df.index)
print('Number of missing days = {}' .format(number_of_missing_dates))


# Check the number of NA's
for colname in df:
    print('Number of NAs in column {} = {}' .format(colname, sum(df[colname].isna())))


# Interpolate NA values (there are very little NA values, thus simple linear interpolation is enough)
df = df.interpolate(method='linear')


# Save the data to a csv
df.to_csv('datasets/weather_data.csv')
# Line to correctly read this csv (for future use)
#df = pd.read_csv('datasets/weather_data.csv', parse_dates=[0], index_col=[0])


# Delete all generated variables
#globals().clear()
