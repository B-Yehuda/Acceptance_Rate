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
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score, log_loss, fbeta_score, recall_score, \
    precision_score
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

def remove_outliers(df, feature: str, threshold: int, above_or_below: str):
    # find index of outliers
    if above_or_below == 'above':
        drop_index = set(df[df[feature] > threshold].index)
    elif above_or_below == 'below':
        drop_index = set(df[df[feature] < threshold].index)
    else:
        raise ValueError('Wrong above_or_below input')

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


def data_processing(df, config):
    # retrieve outliers_to_be_removed from config file
    outliers_to_be_removed = ast.literal_eval(config["DataProcessing"]["outliers_to_be_removed"])

    # retrieve features_to_bucket from config file
    features_to_bucket = ast.literal_eval(config["DataProcessing"]["features_to_bucket"])

    # remove outliers
    for feature, values in outliers_to_be_removed.items():
        df = remove_outliers(df, feature, values['threshold'], values['above_or_below'])

    # convert categorical features to buckets
    for k, v in features_to_bucket.items():
        df = calculate_feature_bucked(df, k, v)

    # drop columns
    col_to_drop = []
    df = df.drop(col_to_drop, axis=1)

    # convert numeric features to category
    numeric_to_category = []
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
    res = {}

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
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
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


# LOSS FUNCTIONS #

class AveragePrecisionScore:
    def __init__(self, y_test, direction):
        self.y_test = y_test
        self.direction = direction
        self.name = "average_precision_score"

    def score(self, y_pred):
        return average_precision_score(self.y_test, y_pred)

class F1Score:
    def __init__(self, y_test, beta_value, direction):
        self.y_test = y_test
        self.beta_value = 1.0
        self.direction = direction
        self.name = "f1_score"

    def score(self, y_pred):
        return fbeta_score(self.y_test, y_pred, beta=self.beta_value)

class RecallScore:
    def __init__(self, y_test, beta_value, direction):
        self.y_test = y_test
        self.beta_value = 2.0
        self.direction = direction
        self.name = "recall_score"

    def score(self, y_pred):
        return fbeta_score(self.y_test, y_pred, beta=self.beta_value)

class PrecisionScore:
    def __init__(self, y_test, beta_value, direction):
        self.y_test = y_test
        self.beta_value = 0.5
        self.direction = direction
        self.name = "precision_score"

    def score(self, y_pred):
        return fbeta_score(self.y_test, y_pred, beta=self.beta_value)

class LogLoss:
    def __init__(self, y_test, direction):
        self.y_test = y_test
        self.direction = direction
        self.name = "log_loss"

    def score(self, y_pred):
        return log_loss(self.y_test, y_pred)


class RMSE:
    def __init__(self, y_test, squared, direction):
        self.y_test = y_test
        self.squared = False
        self.direction = direction
        self.name = "mean_squared_error"

    def score(self, y_pred):
        return mean_squared_error(self.y_test, y_pred, squared=self.squared)


# MODEL CREATION FUNCTIONS #

def tune_models_hyperparams(eval_model, param, X_train, y_train, X_test, y_test):
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
        study = optuna.create_study(sampler=sampler, direction=score_obj.direction)

        # run the study object
        study.optimize(lambda trial: objective(trial,
                                               eval_model,
                                               param,
                                               score_obj.score,
                                               X_train,
                                               y_train,
                                               X_test,
                                               y_test),  # make smart guesses where the best values hyperparameters
                       n_trials=1,  # try hyperparameters combinations n_trials times
                       callbacks=[callback])  # callback save the best model

        # name the best model
        model_name = "regressor_model_" + score_obj.name if eval_model == xgb.XGBRegressor else "classifier_model_" + score_obj.name
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
        # filter models with negative R^2 (models worse than a constant function that predicts the mean of the data)
        filtered_models_grid = {k: v for k, v in grid.items() if v["model_scores"]['R2'] > 0}

        # return best models
        if bool(filtered_models_grid):
            min_log_loss_model = min(filtered_models_grid.values(), key=lambda x: x["model_scores"]['Log Loss'])
            min_log_loss_value = min_log_loss_model["model_scores"]['Log Loss']
            best_models_grid = {k: v for k, v in filtered_models_grid.items() if v["model_scores"]['Log Loss'] == min_log_loss_value}
            return best_models_grid
        else:
            raise ValueError("\033[1m" + 'No model has fitted the data well.' + "\033[0m")

    else:
        # filter models with precision<10%
        best_models_grid = {k: v for k, v in grid.items() if v["model_scores"]['Precision'] > 0.1}

        # return best models
        if bool(best_models_grid):
            return best_models_grid
        else:
            raise ValueError("\033[1m" + 'No model has fitted the data well.' + "\033[0m")


def save_best_model_in_gcs_bucket(best_models_grid, config):
    # retrieve bucket_name from config file
    bucket_name = ast.literal_eval(config["GCS"]["bucket_name"])

    # save best models
    for best_model_item in best_models_grid.items():

        # retrieve best models data
        best_model_name = best_model_item[0]
        best_model_object = best_model_item[1]["model_object"]

        # open (and close) a file where we store the best model
        filename = str(best_model_name) + "_" + str(datetime.now().strftime("%Y%m%d_%H%M")) + ".pkl"
        with open(filename, 'wb') as f:
            # dump best model to the file
            pickle.dump(best_model_object, f)

        # take that file and upload into GCS
        wi_credentials = compute_engine.Credentials()
        storage_client = storage.Client(credentials=wi_credentials)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_filename(filename)


# EXECUTION FUNCTIONS #

def main():
    # # allow GPU memory allocation
    # tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)

    # read from config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # connect redshift
    cur = connect_redshift(config)

    # load data from redshift
    df = load_data(cur, config)

    # data processing
    df = data_processing(df, config)

    # data split
    X_train, X_test, y_train, y_test = data_split(df)

    # define model to pass
    eval_model = xgb.XGBClassifier

    # save models in a dictionary
    param = {"objective": "binary:logistic",  # logistic regression for binary classification, output probability
             "missing": np.nan,  # whenever a null values is encountered it is treated as missing value
             "seed": 42,  # used to generate the folds
             # "tree_method": "gpu_hist",  # speed up processing by using gpu power
             "early_stopping_rounds": 50,  # overfitting prevention, stop early if no improvement in learning
             "eval_metric": "aucpr",  # evaluation metric for validation data
             "n_estimators": 10000  # number of trees
             }
    grid = tune_models_hyperparams(eval_model, param, X_train, y_train, X_test, y_test)

    # print models
    print_grid_results(grid)

    # extract best model
    best_models_grid = get_best_models(eval_model, grid)

    # save best model in a pickle file and upload to gcs bucket
    save_best_model_in_gcs_bucket(best_models_grid, config)


# RUN #

main()
