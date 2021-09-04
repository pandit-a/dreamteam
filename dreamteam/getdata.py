import pandas as pd

"will need changing if files uploaded to GCP or AWS bucket"

file = r'/Users/anand/Documents/Sleep/WSC - variable cross-check_sparse.xlsx'
data = r'/Users/anand/Documents/Sleep/wsc-dataset-0.2.0.csv'

def get_data():
    """
    returns
    """
    df = pd.read_excel(file)
    data_df = pd.read_csv(data)
    return df, data_df


# def remove_columns(df, test=False):
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


if __name__ == '__main__':
    df = get_data()
