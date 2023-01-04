import ast
import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from google.cloud import storage
from google.auth import compute_engine
from datetime import datetime
import gs_chunked_io as gscio
from scipy.sparse import csr_matrix


# IMPORT DATA #

def connect_redshift(config):
    # retrieve credentials from config file
    credentials = ast.literal_eval(config["Redshift_Credentials"]["credentials"])

    # connect redshift
    connection = psycopg2.connect(
        host=credentials['host'],
        port=credentials['port'],
        dbname=credentials['dbname'],
        user=credentials['user'],
        password=credentials['password'])

    # initialize cursor objects
    cur = connection.cursor()

    return cur


def load_data(cur, config):
    # retrieve file name from config file
    filename = config["Dataset"]["filename"]

    # retrieve data set location from config file
    location = config["Dataset"]["location"]

    if location == "REDSHIFT":
        print(f"Loading data from REDSHIFT - started at: \033[1m{datetime.now()}\033[0m")
        # retrieve query from config file
        query = config["Redshift_Data"]["query"]
        # execute query
        cur.execute(query)
        # load the dataset from redshift
        data = cur.fetchall()
        print(f"Loading data from REDSHIFT - finished at: \033[1m{datetime.now()}\033[0m")
        # frame the dataset
        df = pd.DataFrame(data)
        df.columns = [desc[0] for desc in cur.description]
        print(f"Framing the data - finished at: \033[1m{datetime.now()}\033[0m")

    elif location == "GCS":
        print(f"Loading data from GCS - started at: \033[1m{datetime.now()}\033[0m")
        # retrieve ID of GCS bucket from config file
        bucket_name = config["GCS"]["bucket_name"]
        # load the dataset from GCS
        wi_credentials = compute_engine.Credentials()
        storage_client = storage.Client(credentials=wi_credentials)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        # read GCS object (dataset file) in chunks
        with open(filename, "wb") as f:
            for chunk in gscio.for_each_chunk(blob, 25 * 1024 * 1024):
                f.write(chunk)
        df = pd.read_pickle(filename)
        print(f"Loading data from GCS - finished at: \033[1m{datetime.now()}\033[0m")

    elif location == "LOCAL":
        print(f"Loading data from LOCAL FILE - started at: \033[1m{datetime.now()}\033[0m")
        # load the dataset from local dir
        df = pd.read_pickle(filename)
        print(f"Loading data from LOCAL FILE - finished at: \033[1m{datetime.now()}\033[0m")

    else:
        raise ValueError("\033[1m No dataset location was specified in the config file \033[0m")

    return df


# DATA PROCESSING #

def reduce_memory_usage(df, int_cast=True, float_cast=False, obj_to_category=False):
    # check that the df columns are unique
    assert len(df.columns) == len(set(df.columns))

    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    :param df: dataframe to reduce (pd.DataFrame)
    :param int_cast: indicate if columns should be tried to be casted to int (bool)
    :param float_cast: indicate if columns should be tried to be casted to float (bool)
    :param obj_to_category: convert non-datetime related objects to category dtype (bool)
    :return: dataset with the column dtypes adjusted (pd.DataFrame)
    """
    cols = df.columns.tolist()

    for col in cols:
        col_type = df[col].dtypes

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()

            # test column type
            is_int_column = col_type == np.int64
            is_float_column = col_type == np.float64

            if int_cast and is_int_column:
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype('int8')
                elif c_min >= np.iinfo(np.uint8).min and c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype('uint8')
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype('int16')
                elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                    df[col] = df[col].astype('uint16')
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
                elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                    df[col] = df[col].astype('uint32')
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype('int64')
                elif c_min >= np.iinfo(np.uint64).min and c_max <= np.iinfo(np.uint64).max:
                    df[col] = df[col].astype('uint64')
            elif float_cast and is_float_column:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype('float16')
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('float64')
        elif 'datetime' not in col_type.name and obj_to_category:
            df[col] = df[col].astype('category')

    return df


def remove_outliers(df, feature: str, threshold: int, is_above):
    # find index of outliers
    if is_above:
        drop_index = set(df[df[feature] > threshold].index)
    else:
        drop_index = set(df[df[feature] < threshold].index)

    # get index of df without outliers
    new_index = list(set(df.index) - set(drop_index))

    # remove outliers
    df = df.iloc[new_index]

    # reset index
    df.reset_index(drop=True, inplace=True)

    return df


def feature_top_x(df, feature, top_x=None):
    # get the top X most common values of a feature
    data = df[feature].value_counts(normalize=True).head(top_x).to_frame().reset_index()
    top_x_data = list(data.iloc[:, 0])

    return top_x_data


def feature_bucket_based_on_top_x(row, feature, top_x_list):
    # create X buckets of a feature
    if pd.isnull(row[feature]):
        return None
    elif row[feature] in top_x_list:
        return row[feature]
    else:
        return 'Other'


def calculate_feature_bucked(df, feature, top_x):
    # create top x list that contains buckets of a feature
    top_x_list = feature_top_x(df, feature, top_x)
    # create function that will apply the relevant bucket for each row in the data set
    bucket_func = lambda row: feature_bucket_based_on_top_x(row, feature, top_x_list)
    # create a new column that will contain the bucked rows
    bucket_col_name = f"{feature}_bucket"
    # apply function
    bucket_col_data = df.apply(bucket_func, axis=1)
    # add the new column to the df and collect garbage
    df = df.assign(**{bucket_col_name: bucket_col_data.values})
    del bucket_col_data

    return df


def pre_training_data_processing(df, config):
    # retrieve outliers_to_be_removed from config file
    outliers_to_be_removed = ast.literal_eval(config["Data_Processing"]["outliers_to_be_removed"])

    # retrieve features_to_bucket from config file
    features_to_bucket = ast.literal_eval(config["Data_Processing"]["features_to_bucket"])

    # retrieve cols_to_drop from config file
    cols_to_drop = ast.literal_eval(config["Data_Processing"]["cols_to_drop"])

    # retrieve numeric features to convert to categorical from config file
    numeric_to_category = ast.literal_eval(config["Data_Processing"]["numeric_to_category"])

    # remove outliers
    for feature, value in outliers_to_be_removed.items():
        df = remove_outliers(df, feature, value['threshold'], value['is_above'])

    # convert categorical features to buckets
    for feature, bucket in features_to_bucket.items():
        df = calculate_feature_bucked(df, feature, bucket)

    # drop columns
    df = df.drop(cols_to_drop, axis=1)

    # convert numeric features to category
    for feature in numeric_to_category:
        df[feature] = df[feature].astype(object)

    return df


def data_split(df):
    # declare feature vector and target variable
    X = df.drop('is_accept', axis=1)
    y = df['is_accept']

    # create dummy variables
    X = pd.get_dummies(X)

    # # sparse the data
    # X = csr_matrix(X)

    # split the dataset
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)