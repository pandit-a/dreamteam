import pandas as pd

"will need changing if files uploaded to GCP or AWS bucket"

file = r'/Users/anand/Documents/Sleep/WSC - variable cross-check_sparse.xlsx'
data = r'/Users/anand/Documents/Sleep/wsc-dataset-0.2.0.csv'

def get_data():
    """
    returns df/excel indicating which columns to keep and data_df/csv which is main data frame
    """
    df = pd.read_excel(file)
    data_df = pd.read_csv(data)
    return df, data_df


def clean_data(df, data_df, duplicates = True):
    deleted = df[df['Proposed Removal'] == 'R']
    deleted_cols = deleted.iloc[:, 0]
    data_df = data_df.drop(deleted_cols.to_list(), axis=1)
    data_df.set_index('wsc_id', inplace=True)

    if duplicates = False:
        data_df.drop_duplicates('wsc_id', inplace=True)

    data_df.nasal_cong_none.replace({np.nan:0,'Y':1}, inplace=True)
    data_df.num_pregnancies.replace({np.nan:0}, inplace=True)
    data_df.packs_week.replace({np.nan:0}, inplace=True)
    data_df.pack_years.replace({np.nan:0}, inplace=True)

    return data_df

#     df = df.dropna(how='any', axis='rows')
#     df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
#     df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
#     if "fare_amount" in list(df):
#         df = df[df.fare_amount.between(0, 4000)]
#     df = df[df.passenger_count < 8]
#     df = df[df.passenger_count >= 0]
#     df = df[df["pickup_latitude"].between(left=40, right=42)]
#     df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
#     df = df[df["dropoff_latitude"].between(left=40, right=42)]
#     df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
#     return df


# if __name__ == '__main__':
#     df = get_data()
