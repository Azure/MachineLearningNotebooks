def transform(df, index):
    df['Latitude'].fillna('0', inplace=True)
    df['Longitude'].fillna('0', inplace=True)
    return df
