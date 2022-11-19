import pandas as pd
import numpy as np

def extract_lobster_data():
    print("\nExtracting Lobster Data from: https://www.maine.gov/dmr/fisheries/commercial/landings-data")
    lobster_source_data = pd.read_csv("data/lobster.table.csv")

    # Extract needed data from summarized lobster reported data
    year_all           = lobster_source_data['YEAR'].tolist()
    lobster_pounds_all = lobster_source_data['POUNDS'].tolist()
    celcius_all        = lobster_source_data['STATION (C)'].tolist()
    value_millions_all = lobster_source_data['VALUE(millions)'].tolist()

    # get the index for the years i would like to limit the data with
    index_lobster_start = year_all.index(1989)
    index_lobster_end   = year_all.index(2019) + 1

    # limit data to desired years
    year_1989_2019                   = year_all[index_lobster_start:index_lobster_end]
    lobster_pounds_1989_2019         = [int(p.replace(',', '')) for p in lobster_pounds_all[index_lobster_start:index_lobster_end]]
    lobster_pounds_1989_2019_million = [p/1000000 for p in lobster_pounds_1989_2019]
    celcius_all_1989_2019            = celcius_all[index_lobster_start:index_lobster_end]

    return year_1989_2019, lobster_pounds_1989_2019_million, celcius_all_1989_2019

def extract_cei_data():
    print("\nExtracting Extreme weather Data from: https://www.ncei.noaa.gov/access/monitoring/cei/ ")
    extreme_source_data = pd.read_csv("data/cei.csv")

    # Extract needed data from climate extreme index data
    year_cei_all     = extreme_source_data['Date'].tolist()
    percent_area_all = extreme_source_data['Percentage Area'].tolist()

    # get the index for the years i would like to limit the data with
    index_cei_start = year_cei_all.index(1989)
    index_cei_end   = year_cei_all.index(2019) + 1

    percent_area_1989_2019 = percent_area_all[index_cei_start:index_cei_end]

    return percent_area_1989_2019

def extract_sea_surface_temperature_data():
    print("\nExtracting Sea Surface temperature Data from: https://www.maine.gov/dmr/science/weather-tides/boothbay-harbor-environmental-data")
    sea_temperature_data = pd.read_csv("data/1905-2019sst.csv")

    # Extract needed data from surface sea temperature data
    date_sea_all    = sea_temperature_data['COLLECTION_DATE'].tolist()
    max_sea_all     = sea_temperature_data['Sea Surface Temp Max C'].tolist()
    min_sea_all     = sea_temperature_data['Sea Surface Temp Min C'].tolist()
    average_sea_all = sea_temperature_data['Sea Surface Temp Ave C'].tolist()

    # wrangle sea temperature as its reported daily
    sea_df = pd.DataFrame(list(zip(date_sea_all, max_sea_all, min_sea_all, average_sea_all)),
                columns =['Date', 'Max', 'Min', 'Avg'])

    # remove rows with blank fields
    sea_df.dropna(
        axis=0,
        how='any',
        inplace=True
    )

    sea_df['Year'] = sea_df['Date'].apply(lambda date: date.split('/')[-1])

    # reducing all data to average by year
    df_sea_average_by_year = sea_df.groupby('Year').agg(
        Mean=('Avg', np.median),
        Min=('Min', np.median),
        Max=('Max', np.median)
    )

    df_sea_average_by_year['Range'] = df_sea_average_by_year['Max'] - df_sea_average_by_year['Min']
    print('\nDataframe average by year')
    print(df_sea_average_by_year)

    # reducing all data to average by year
    df_sea_by_year = sea_df.groupby('Year').agg(
        Mean=('Avg', np.mean),
        Min=('Min', np.min),
        Max=('Max', np.max)
    )

    df_sea_by_year['Range'] = df_sea_by_year['Max'] - df_sea_by_year['Min']
    print('\nDataframe by year')
    print(df_sea_by_year)

    # Data already between desired dates

    return df_sea_average_by_year['Max'].tolist(), df_sea_average_by_year['Min'].tolist(), df_sea_average_by_year['Mean'].tolist(), df_sea_by_year['Max'].tolist(), df_sea_by_year['Min'].tolist(), df_sea_by_year['Range'].tolist()
