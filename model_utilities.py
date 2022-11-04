import pandas as pd
import pickle
import numpy as np


def load_model_file():
    # load the model from disk
    filename = 'model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))

    # extract all the columns (with order) - from training dataset
    cols_when_model_builds = loaded_model.feature_names_in_

    return loaded_model, cols_when_model_builds


def categorical_features_to_bucket():
    # categorical features to bucket
    features = ['most_played_game', 'country', 'language']

    return features


def retrieve_model_buckets_data():
    # pass categorical features to bucket
    features_to_bucket = categorical_features_to_bucket()

    # pass loaded model columns
    cols_when_model_builds = load_model_file()[1]

    # save buckets columns - from training dataset
    model_buckets_data = {}
    for feature in features_to_bucket:
        model_buckets_data[feature] = [col.replace(f"{feature}_bucket_", '') for col in cols_when_model_builds if
                                       f"{feature}" in col]
    return model_buckets_data


def bucket_df_rows(row, feature, model_bucket_values):
    if pd.isnull(row[feature]):
        return np.nan
    elif row[feature] in model_bucket_values:
        return row[feature]
    else:
        return 'Other'


def df_apply_bucket_transformation(df):
    # pass categorical features to bucket
    features_to_bucket = categorical_features_to_bucket()

    # retrieve buckets columns - from training dataset
    model_buckets_data = retrieve_model_buckets_data()

    # for every feature that should be bucketed - bucket it according to model buckets
    for feature in features_to_bucket:
        model_bucket_values = model_buckets_data[feature]
        df[feature] = df.apply(lambda row: bucket_df_rows(row, feature, model_bucket_values), axis=1)
        df = df.rename(columns={f"{feature}": f"{feature}_bucket"})

    return df


def data_processing(df):
    # pass loaded model columns
    cols_when_model_builds = load_model_file()[1]

    # bucket categorical features of the df (curation list)
    df = df_apply_bucket_transformation(df)

    # drop columns
    col_to_drop = ['channel_id']
    df = df.drop(col_to_drop, axis=1)

    # convert numeric features to categorical
    numeric_to_category = ['is_tipping_panel', 'is_bot_command_usage',
                           'is_overlay', 'is_website_visit',
                           'is_se_live', 'is_alert_box_fired',
                           'is_open_stream_report']
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


def predict_pipeline(df):
    # pass loaded model
    loaded_model = load_model_file()[0]

    # store channel_id before data processing
    channels = df['channel_id']

    # data processing
    df = data_processing(df)

    # predict with the model
    df['predicted_acceptance_rate'] = loaded_model.predict(df)
    df = df.join(channels)
    df = df[['channel_id', 'predicted_acceptance_rate']]

    return df


def create_predictions_df(df):
    # import predictions
    df = predict_pipeline(df)

    # define acceptance_rate column
    col = 'predicted_acceptance_rate'

    # create acceptance_rate segment
    conditions = [
        df[col].between(0, 0.5),
        df[col].between(0.5, 0.75),
        df[col].between(0.75, 1)]
    choices = ["Tier_3", "Tier_2", "Tier_1"]
    df["segment"] = np.select(conditions, choices, default=np.nan)

    # sort
    df = df.sort_values(['predicted_acceptance_rate'], ascending=[False])

    return df
