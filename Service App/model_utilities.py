import ast
import configparser
import pandas as pd
import pickle
import numpy as np
from google.cloud import storage
from google.auth import compute_engine


def read_from_gcs_bucket(config):
    # retrieve ID of GCS bucket from config file
    bucket_name = ast.literal_eval(config["GCS"]["bucket_name"])

    # retrieve ID of GCS object from config file
    filename = ast.literal_eval(config["GCS"]["filename"])

    # load the model (pkl file) from GCS
    wi_credentials = compute_engine.Credentials()
    storage_client = storage.Client(credentials=wi_credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.download_to_filename(filename)
    loaded_model = pickle.load(open(filename, 'rb'))

    # extract all the columns (with order) - from training dataset
    cols_when_model_builds = loaded_model.feature_names_in_

    return loaded_model, cols_when_model_builds


def load_model_file(config):
    # load the model (pkl file)  from disk
    filename = ast.literal_eval(config["GCS"]["filename"])
    loaded_model = pickle.load(open(filename, 'rb'))

    # extract all the columns (with order) - from training dataset
    cols_when_model_builds = loaded_model.feature_names_in_

    return loaded_model, cols_when_model_builds


def retrieve_model_bucketed_columns(cols_when_model_builds, config):
    # retrieve categorical features to bucket from config filet
    features_to_bucket = ast.literal_eval(config["DataProcessing"]["features_to_bucket"])

    # save buckets columns - from training dataset
    model_bucketed_columns = {}
    for feature in features_to_bucket:
        model_bucketed_columns[feature] = [col.replace(f"{feature}_bucket_", '') for col in cols_when_model_builds if
                                           f"{feature}" in col]

    return model_bucketed_columns


def bucket_df_rows(row, feature, model_bucketed_columns_values):
    if pd.isnull(row[feature]):
        return np.nan
    elif row[feature] in model_bucketed_columns_values:
        return row[feature]
    else:
        return 'Other'


def df_apply_bucket_transformation(df, model_bucketed_columns, config):
    # retrieve categorical features to bucket from config filet
    features_to_bucket = ast.literal_eval(config["DataProcessing"]["features_to_bucket"])

    # for every feature that should be bucketed - bucket it according to model buckets
    for feature in features_to_bucket:
        model_bucketed_columns_values = model_bucketed_columns[feature]
        df[feature] = df.apply(lambda row: bucket_df_rows(row, feature, model_bucketed_columns_values), axis=1)
        df = df.rename(columns={f"{feature}": f"{feature}_bucket"})

    return df


def data_processing(df, cols_when_model_builds, config):
    # drop columns
    col_to_drop = ['channel_id']
    df = df.drop(col_to_drop, axis=1)

    # retrieve numeric features to convert to categorical from config filet
    numeric_to_category = ast.literal_eval(config["DataProcessing"]["numeric_to_category"])

    # convert numeric features to categorical
    for feature in numeric_to_category:
        df[feature] = df[feature].astype(object)

    # create dummy variables
    df = pd.get_dummies(df)

    # add missing columns (features) from training dataset - to the imported data (csv file)
    for col in cols_when_model_builds:
        if col not in list(df):
            df[col] = 0

    # reorder the imported data (csv file) columns - to match the training dataset columns
    df = df[cols_when_model_builds]

    return df


def create_predictions_df(df):
    # read config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # load model
    loaded_model, cols_when_model_builds = read_from_gcs_bucket(config)

    # store channel_id before data processing
    channels = df['channel_id']

    # retrieve model bucketed columns (similar to columns from training dataset)
    model_bucketed_columns = retrieve_model_bucketed_columns(cols_when_model_builds, config)

    # bucket categorical features of the df (csv file)
    df = df_apply_bucket_transformation(df, model_bucketed_columns, config)

    # data processing
    df = data_processing(df, cols_when_model_builds, config)

    # predict with the model
    df['predicted_acceptance_rate'] = loaded_model.predict(df)
    df = df.join(channels)
    df = df[['channel_id', 'predicted_acceptance_rate']]

    # define acceptance_rate column
    col = 'predicted_acceptance_rate'

    # create acceptance_rate segment
    conditions = [
        df[col].between(0, 0.25),
        df[col].between(0.25, 0.5),
        df[col].between(0.5, 0.75),
        df[col].between(0.75, 1)]
    choices = ["Tier_4", "Tier_3", "Tier_2", "Tier_1"]
    df["segment"] = np.select(conditions, choices, default=np.nan)

    # sort
    df = df.sort_values(['predicted_acceptance_rate'], ascending=[False])

    return df
