# let us import the data first

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns

# change these according to pwd
AIRPORT_DATA = 'data/case_ds.csv'
IMG_FOLDER = 'img'

def import_data(path=AIRPORT_DATA):
    ''' imports and adds some temporal variables based on the date string '''
    df = pd.read_csv(path, delimiter=';')
    df['real_date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.set_index('Date', inplace=True)
    total_rows = len(df)
    print("Total rows:", total_rows)

    # Extract components from the datetime column
    df['Year'] = df['real_date'].dt.year
    df['Month'] = df['real_date'].dt.month
    df['Day'] = df['real_date'].dt.day
    df['Week'] = df['real_date'].dt.isocalendar().week   #df['Week'] = df['real_date'].dt.strftime('%W') 
    df['DayOfWeek'] = df['real_date'].dt.dayofweek
    df['DayName'] = df['real_date'].dt.day_name()   
        
    # Printing min and max of seats
    print("Min of seats:", df['Seats'].min())
    print("Max of seats:", df['Seats'].max())

    # cleaning up / data rensning: 
    filtered_rows = df[df['PAX'] - df['Seats'] >= 0]
    print("Num of rows where pax will be adjusted to seats", len(filtered_rows))
    df.loc[df['PAX'] - df['Seats'] >= 1, 'PAX'] = df.loc[df['PAX'] - df['Seats'] >= 1, 'Seats'].values
    
    return df


def basic_feature_engineering(df):
    """ make new features """
    # Percentage occupancy per day
    df['PCT_occupied'] = (df['PAX'] / df['Seats']) * 100

    # Printing unique nominal values 
    print("Unique status values:", df['Status'].unique())
    print("Number of unique statuses:", df['Status'].nunique())
    print("Unique route values:", df['Route'].nunique())
    print("Number of unique airports:", df['Airport'].nunique()) 
    print("Der er naturligvis høj correlation mellem airport og routes: ", df.groupby(['Airport', 'Route']).ngroups)
    
    # Modelling average percentage occupancy per week as a feature to capture seasonal trends
    df['Avg_PCT_occupied_weekly'] = (
    df.groupby(['Route', 'Airport', 'Week'])['PCT_occupied']
    .transform('mean'))

    return df


def summary_statistics(df):

    # making some box plots to capture variance
    df[['Seats', 'PAX']].plot(kind='box')
    plt.title('Box Plot of seats and passengers')
    plt.ylabel('Number of seats')
    plt.savefig(os.path.join(IMG_FOLDER,'boxplot-seats-pax.png'))
    #plt.show()

    # Find the most popular 3 airport, route pairs
    pair_counts = df.groupby(['Route', 'Airport']).size().reset_index(name='count')
    most_common_pairs = pair_counts.sort_values(by='count', ascending=False).head(3)
    print(most_common_pairs)
    top_pairs = set(
        tuple(x) for x in most_common_pairs[['Route', 'Airport']].head(3).values
    )

    # show seasonality in the top 3 airport, routes
    df_top = df[df[['Route', 'Airport']].apply(tuple, axis=1).isin(top_pairs)]
    df_grouped = (
        df_top
        .groupby(['Route', 'Airport', 'Week'], as_index=False)['Avg_PCT_occupied_weekly']
        .first()
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df_grouped,
        x='Week',
        y='Avg_PCT_occupied_weekly',
        hue='Route',
        style='Airport',
        markers=True,
        dashes=False
    )

    plt.title('Seasonality in the occupancy of flights(Top 3 Route/Airport Pairs)')
    plt.ylabel('Average pct. occupied seats')
    plt.xlabel('Week Number')
    plt.grid(True)
    plt.legend(title='Route / Airport', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_FOLDER,'seasonality-occupancy.png'))
    plt.show()

    return df


def adv_feature_engineering(df):
    ''' other features added here and redundant ones removed '''
    # the current existing columns
    print("the current columns are: ", df.columns)
    
    # the obvious one hot encoding of Status
    # One-hot encode status column
    status_dummies = pd.get_dummies(df['Status'], prefix='status').astype(int)

    # Concatenate back to the original DataFrame
    df = pd.concat([df, status_dummies], axis=1)

    # compute 3 weeks rolling average for occupancy mean
    weekly_avg = (
        df.groupby(['Route', 'Airport', 'Year', 'Week'], as_index=False)['PCT_occupied']
        .mean()
        .rename(columns={'PCT_occupied': 'Weekly_PCT_occupied'}))
    weekly_avg.sort_values(['Route', 'Airport', 'Year', 'Week'], inplace=True)
    weekly_avg['Rolling_PCT_occupied_3w'] = (
        weekly_avg
        .groupby(['Route', 'Airport'])['Weekly_PCT_occupied']
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(drop=True))
    weekly_avg['Rolling_PCT_occupied_3w'].fillna(weekly_avg['Weekly_PCT_occupied'])
    original_index = df.index
    df = df.merge(weekly_avg[['Route', 'Airport', 'Year', 'Week', 'Rolling_PCT_occupied_3w']],
                on=['Route', 'Airport', 'Year', 'Week'],
                how='left')
    df.index = original_index
    print(df)

    df['flight_id'] = df['Route'].astype(str) + '__' + df['Airport'].astype(str)
    # aggregating 
    daily_agg = (
        df.groupby('Date').agg(
            PAX=('PAX', 'sum'),
            num_flights=('flight_id', 'nunique'),
            Avg_Rolling_PCT_3w=('Rolling_PCT_occupied_3w', 'mean'),
            Seats_total=('Seats', 'sum'),
            Week=('Week', 'first'),
            DayOfWeek=('DayOfWeek', 'first'),
            Year=('Year', 'first'),
        ) 
    ).reset_index()

    # List one-hot columns explicitly or detect them
    status_cols = [col for col in df.columns if col.startswith('status_')]
    daily_status_agg = df.groupby('Date')[status_cols].sum().reset_index()

    flight_agg_df = pd.merge(daily_agg, daily_status_agg, on='Date', how='inner')
    print("Feature Engineering phase done: \n", flight_agg_df.drop('Date', axis=1))
    return flight_agg_df


def preprocessing():
    df = import_data()
    basic_feature_engineering(df)
    summary_statistics(df)
    flight_agg_df = adv_feature_engineering(df)
    return flight_agg_df

if __name__ == '__main__':
    preprocessing()