# import os
# import resource
import ast
import configparser
import psycopg2
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
# import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, log_loss, fbeta_score, recall_score, precision_score
from loss_functions import AveragePrecisionScore, F1Score, RecallScore, PrecisionScore, LogLoss, RMSE
import optuna
from optuna.samplers import TPESampler
from google.cloud import storage
from google.auth import compute_engine


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
    # retrieve query from config file
    query = config["Redshift_Data"]["query"]

    # load data
    cur.execute(query)

    # frame the data
    df = pd.DataFrame(cur.fetchall())
    df.columns = [desc[0] for desc in cur.description]

    return df


# DATA PROCESSING #

def reduce_memory_usage(df):
    # check that the df columns are unique
    assert len(df.columns) == len(set(df.columns))

    # reduce memory usage of Python objects
    for col in df.columns:
        if df[col].dtypes == np.int64:
            df[col] = df[col].astype(int)
        elif df[col].dtypes == np.float64:
            df[col] = df[col].astype('float32')

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
    # apply the relevant bucket for each sample in the data set
    top_x_list = feature_top_x(df, feature, top_x)
    df[f"{feature}_bucket"] = df.apply(lambda row: feature_bucket_based_on_top_x(row, feature, top_x_list), axis=1)
    return df


def pre_training_data_processing(df, config):
    # retrieve outliers_to_be_removed from config file
    outliers_to_be_removed = ast.literal_eval(config["DataProcessing"]["outliers_to_be_removed"])

    # retrieve features_to_bucket from config file
    features_to_bucket = ast.literal_eval(config["DataProcessing"]["features_to_bucket"])

    # retrieve cols_to_drop from config file
    cols_to_drop = ast.literal_eval(config["DataProcessing"]["cols_to_drop"])

    # retrieve numeric features to convert to categorical from config file
    numeric_to_category = ast.literal_eval(config["DataProcessing"]["numeric_to_category"])

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
    # Declare feature vector and target variable
    X = df.drop('is_accept', axis=1)
    y = df['is_accept']

    # Create dummy variables
    X = pd.get_dummies(X)

    # split the dataset
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# MODELS EVALUATION #

def model_performance(eval_model, model_object, X_test, y_test):

    # predict with the model
    y_pred = model_object.predict(X_test)

    if eval_model == xgb.XGBRegressor:

        # regression model scoring
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        logistic_loss = log_loss(y_test, y_pred)

        # regression scores
        regression_res = {'RMSE': rmse, 'R2': r2, 'Log Loss': logistic_loss}

        return regression_res

    else:

        # classification model scoring
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        fb = fbeta_score(y_test, y_pred, beta=2.0)

        # classification scores
        classification_res = {'Precision': precision, 'Recall': recall, 'F_beta': fb}

        return classification_res


# HYPERPARAMETERS OPTIMIZATION #

# disable printing
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, eval_model, param, score_func, X_train, y_train, X_test, y_test):
    # create objective function which evaluate model performance based on the hyperparameters combination

    # hyperparameters to be tuned
    hyperparameters_candidates = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6),
        "subsample": trial.suggest_float("subsample", 0.4, 0.8),
        "alpha": trial.suggest_float("alpha", 0.01, 10.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_float("lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000),
        "scale_pos_weight": trial.suggest_int('scale_pos_weight', 1, 100)
    }

    # instantiate the model
    xgb_optuna = eval_model(**param, **hyperparameters_candidates)

    # Add a callback for pruning (ensure unpromising trials are stopped early)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-aucpr")

    # fit
    xgb_optuna.fit(X_train,
                   y_train,
                   verbose=False,
                   eval_set=[(X_test, y_test)],
                   callbacks=[pruning_callback])

    # tag each trial with keyword
    trial.set_user_attr(key="model", value=xgb_optuna)

    # predict with the model
    y_pred = xgb_optuna.predict(X_test)

    # score the model
    score = score_func(y_pred)

    return score


def callback(study, trial):
    # create a callback function to retrieve the best model (i.e. the model with the best hyperparameters)
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])


# MODEL CREATION FUNCTIONS #

def tune_models_hyperparams(eval_model, X_train, y_train, X_test, y_test, config):
    # retrieve param from config file
    param = ast.literal_eval(config["ModelParameters"]["param"])
    param["missing"] = np.nan  # whenever a null value is encountered it is treated as missing value

    # retrieve n_trials from config file
    n_trials = int(config["ModelParameters"]["n_trials"])

    # dictionary for saving models
    grid = {}

    # define scoring method
    if eval_model == xgb.XGBRegressor:
        scoring_objects = [AveragePrecisionScore(y_test, direction="maximize"),
                           LogLoss(y_test, direction="minimize"),
                           RMSE(y_test, squared=False, direction="minimize")
                           ]
    else:
        scoring_objects = [AveragePrecisionScore(y_test, direction="maximize"),
                           F1Score(y_test, beta_value=1.0, direction="maximize"),
                           RecallScore(y_test, beta_value=2.0, direction="maximize"),
                           PrecisionScore(y_test, beta_value=0.5, direction="maximize")
                           ]

    # create a sampler object to find more efficiently the best hyperparameters
    sampler = TPESampler()  # by default the sampler = TPESampler()

    for score_obj in scoring_objects:
        # create a study object to set the direction of optimization and the sampler
        study = optuna.create_study(sampler=sampler, direction=score_obj.direction, storage="sqlite:///acceptance.db")

        # run the study object
        study.optimize(lambda trial: objective(trial,
                                               eval_model,
                                               param,
                                               score_obj.score,
                                               X_train,
                                               y_train,
                                               X_test,
                                               y_test),  # make smart guesses where the best values hyperparameters
                       n_trials=n_trials,  # try hyperparameters combinations n_trials times
                       callbacks=[callback],  # callback save the best model
                       gc_after_trial=True)  # garbage collector

        # name the best model
        model_name = "regressor_model_" + score_obj.name \
            if eval_model == xgb.XGBRegressor \
            else "classifier_model_" + score_obj.name
        grid[model_name] = {}

        # initiate best model
        model_object = study.user_attrs["best_model"]
        grid[model_name]["model_object"] = model_object

        # score best model
        model_scores = model_performance(eval_model, model_object, X_test, y_test)
        grid[model_name]["model_scores"] = model_scores

    return grid


def print_grid_results(grid):
    # print models results
    for name, model in grid.items():
        print('{:-^70}'.format(' [' + name + '] '))
        print(model["model_scores"])


def get_best_models(eval_model, grid):

    if eval_model == xgb.XGBRegressor:
        # filter out models with negative R2 (models worse than a constant function that predicts the mean of the data)
        filtered_models_grid = {model_name: model_data for model_name, model_data in grid.items()
                                if model_data["model_scores"]['R2'] > 0}

        # return best models
        if bool(filtered_models_grid):
            min_log_loss_model = min(filtered_models_grid.values(), key=lambda x: x["model_scores"]['Log Loss'])
            min_log_loss_value = min_log_loss_model["model_scores"]['Log Loss']
            best_models_grid = {model_name: model_data for model_name, model_data in filtered_models_grid.items()
                                if model_data["model_scores"]['Log Loss'] == min_log_loss_value}
            return best_models_grid
        else:
            raise ValueError("No model has fitted the data well")

    else:
        # filter models with precision<10%
        best_models_grid = {model_name: model_data for model_name, model_data in grid.items()
                            if model_data["model_scores"]['Precision'] > 0.1}

        # return best models
        if bool(best_models_grid):
            return best_models_grid
        else:
            raise ValueError("No model has fitted the data well")


def generate_model_file_name(best_model_name):
    model_file_name = str(best_model_name) + "_" + str(datetime.now().strftime("%Y%m%d_%H%M")) + ".pkl"

    return model_file_name


def create_model_pickle_file(model_file_path, best_model_data):
    # open (and close) a file where we store the best model
    with open(model_file_path, 'wb') as f:
        # dump best model to the file
        pickle.dump(best_model_data["model_object"], f)

    return model_file_path


def upload_to_gcs(model_file_path, config):
    # retrieve bucket_name from config file
    bucket_name = config["GCS"]["bucket_name"]

    # take that file and upload into GCS
    wi_credentials = compute_engine.Credentials()
    storage_client = storage.Client(credentials=wi_credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_file_path)
    blob.upload_from_filename(model_file_path)


# EXECUTION FUNCTIONS #

def main():
    # # check if we are running in a container and set memory limits appropriately
    # if os.path.isfile('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    #     with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as limit:
    #         mem = int(limit.read())
    #         resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    # # allow GPU memory allocation
    # tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)

    # read from config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # connect redshift
    cur = connect_redshift(config)

    # load data from redshift and reduce its memory usage
    df = reduce_memory_usage(load_data(cur, config))

    # data processing
    df = pre_training_data_processing(df, config)

    # data split
    X_train, X_test, y_train, y_test = data_split(df)

    # delete unnecessary objects from memory
    # objects_to_delete = [df, X, y]
    # for obj in objects_to_delete:
    #     del obj
    del df

    # define model type
    eval_model = xgb.XGBClassifier if config["ModelParameters"]["model_type"] == "CLASSIFIER" else xgb.XGBRegressor

    # save models in a dictionary
    grid = tune_models_hyperparams(eval_model, X_train, y_train, X_test, y_test, config)

    # print models
    print_grid_results(grid)

    # extract best models
    best_models_grid = get_best_models(eval_model, grid)

    # save best model in a pickle file and upload to gcs bucket
    for best_model_name, best_model_data in best_models_grid.items():
        model_file_path = generate_model_file_name(best_model_name)
        create_model_pickle_file(model_file_path, best_model_data)
        # upload_to_gcs(model_file_path, config)


# RUN #

if __name__ == "__main__":
    main()
