import os
import ast
import configparser
import psycopg2
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, log_loss, fbeta_score, recall_score, precision_score
from loss_functions import AveragePrecisionScore, F1Score, RecallScore, PrecisionScore, LogLoss, RMSE
import optuna
from optuna.samplers import TPESampler
from google.cloud import storage
from google.auth import compute_engine
from datetime import datetime
import gs_chunked_io as gscio


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
    # apply the relevant bucket for each sample in the data set
    top_x_list = feature_top_x(df, feature, top_x)
    df[f"{feature}_bucket"] = df.apply(lambda row: feature_bucket_based_on_top_x(row, feature, top_x_list), axis=1)
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

    elif eval_model == xgb.XGBClassifier:

        # classification model scoring
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        fb = fbeta_score(y_test, y_pred, beta=2.0)

        # classification scores
        classification_res = {'Precision': precision, 'Recall': recall, 'F_beta': fb}

        return classification_res


# HYPERPARAMETERS OPTIMIZATION #

def objective(trial, eval_model, param, score_func, score_name, X_train, y_train, X_test, y_test):
    # create objective function which evaluate model performance based on the hyperparameters combination

    # hyperparameters to be tuned
    hyperparameters_candidates = {
        "n_estimators": 10000,  # number of trees
        "early_stopping_rounds": 100,  # overfitting prevention, stop early if no improvement in learning
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000),
        "subsample": trial.suggest_float("subsample", 0.4, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),
        "alpha": trial.suggest_float("alpha", 0.01, 10.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_float("lambda", 1e-8, 10.0),
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

    # for each trial - create naming convention
    model_name = "regressor_model" if eval_model == xgb.XGBRegressor else "classifier_model"

    # tag each trial - with its naming convention
    trial.set_user_attr(key="model_name", value=f"{model_name}_{score_name}")

    # for each model object - create a path based on the trial tag
    model_file_path = f"{trial.user_attrs['model_name']}_{trial.number}.pkl"

    # open (and close) a file where we store each model object
    with open(model_file_path, 'wb') as f:
        # dump best model to the file
        pickle.dump(xgb_optuna, f)

    # predict with the model
    y_pred = xgb_optuna.predict(X_test)

    # score the model
    score = score_func(y_pred)

    return score


def callback(study, trial):
    # create a callback function to delete all models besides the best one
    file_prefix = f"{trial.user_attrs.get('model_name', -1)}"
    best_model_file_path = f"{file_prefix}_{study.best_trial.number}.pkl"
    model_files_to_delete = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(file_prefix) and f != best_model_file_path]
    for f in model_files_to_delete:
        os.remove(f)
    print(f"Optuna {trial.user_attrs.get('model_name', -1)} trial number {trial.number} - finished at: \033[1m{datetime.now()}\033[0m")


# MODEL CREATION FUNCTIONS #

def tune_models_hyperparams(eval_model, X_train, y_train, X_test, y_test, config):
    # retrieve param from config file
    param = ast.literal_eval(config["Model_Parameters"]["param"])
    param["missing"] = np.nan  # whenever a null value is encountered it is treated as missing value

    # retrieve n_trials from config file
    n_trials = int(config["Model_Parameters"]["n_trials"])

    # retrieve Scoring_Functions values from config file
    regressor_scoring_functions = ast.literal_eval(config["Scoring_Functions"]["regressor_scoring_functions"])
    classifier_scoring_functions = ast.literal_eval(config["Scoring_Functions"]["classifier_scoring_functions"])

    # dictionary for saving the scoring_functions
    scoring_functions = {}

    # dictionary for saving the best models
    grid = {}

    # define scoring method
    if eval_model == xgb.XGBRegressor:
        for reg_score_obj in regressor_scoring_functions:
            if reg_score_obj == "AveragePrecisionScore":
                scoring_functions[reg_score_obj] = AveragePrecisionScore(y_test, direction="maximize")
            elif reg_score_obj == "LogLoss":
                scoring_functions[reg_score_obj] = LogLoss(y_test, direction="minimize")
            elif reg_score_obj == "RMSE":
                scoring_functions[reg_score_obj] = RMSE(y_test, squared=False, direction="minimize")
            else:
                raise ValueError("\033[1m regressor_scoring_functions object was not defined correctly in the config file \033[0m")

    elif eval_model == xgb.XGBClassifier:
        for clf_score_obj in classifier_scoring_functions:
            if clf_score_obj == "AveragePrecisionScore":
                scoring_functions[clf_score_obj] = AveragePrecisionScore(y_test, direction="maximize")
            elif clf_score_obj == "F1Score":
                scoring_functions[clf_score_obj] = F1Score(y_test, beta_value=1.0, direction="maximize")
            elif clf_score_obj == "RecallScore":
                scoring_functions[clf_score_obj] = RecallScore(y_test, beta_value=2.0, direction="maximize")
            elif clf_score_obj == "PrecisionScore":
                scoring_functions[clf_score_obj] = PrecisionScore(y_test, beta_value=0.5, direction="maximize")
            else:
                raise ValueError("\033[1m classifier_scoring_functions object was not defined correctly in the config file \033[0m")

    # create a sampler object to find more efficiently the best hyperparameters
    sampler = TPESampler()  # by default the sampler = TPESampler()

    for score_obj in scoring_functions.values():
        # create a study object to set the direction of optimization and the sampler
        study = optuna.create_study(sampler=sampler,
                                    direction=score_obj.direction,
                                    storage="sqlite:///study_state.db")

        # run the study object
        study.optimize(lambda trial: objective(trial,
                                               eval_model,
                                               param,
                                               score_obj.score,
                                               score_obj.name,
                                               X_train,
                                               y_train,
                                               X_test,
                                               y_test),  # make smart guesses where the best values hyperparameters
                       n_trials=n_trials,  # try hyperparameters combinations n_trials times
                       callbacks=[callback],  # callback delete all models but the best one
                       gc_after_trial=True)  # garbage collector

        # store best model name
        model_name = "regressor_model_" + score_obj.name \
            if eval_model == xgb.XGBRegressor \
            else "classifier_model_" + score_obj.name
        grid[model_name] = {}

        # store best model path
        model_file_path = f"{model_name}_{study.best_trial.number}.pkl"
        grid[model_name]["model_file_path"] = model_file_path

        # store best model object
        with open(model_file_path, 'rb') as f:
            model_object = pickle.load(f)
        grid[model_name]["model_object"] = model_object

        # store best model score
        model_scores = model_performance(eval_model, model_object, X_test, y_test)
        grid[model_name]["model_scores"] = model_scores

    return grid


def print_grid_results(grid):
    # print models results
    for name, model in grid.items():
        print('{:-^70}'.format(' [' + name + '] '))
        print(model["model_scores"])


def get_best_models(eval_model, grid, config):
    # retrieve Scoring_Functions filters from config file
    r2_filter = float(config["Scoring_Functions"]["r2_filter"])
    precision_filter = float(config["Scoring_Functions"]["precision_filter"])

    # dictionary for saving the filtered best models
    best_models_grid = {}

    if eval_model == xgb.XGBRegressor:
        # filter out models with negative R2 (models worse than a constant function that predicts the mean of the data)
        filtered_models_grid = {model_name: model_data for model_name, model_data in grid.items()
                                if model_data["model_scores"]['R2'] > r2_filter}

        # return best models and delete the rest
        if filtered_models_grid:
            min_log_loss_model = min(filtered_models_grid.values(), key=lambda x: x["model_scores"]['Log Loss'])
            min_log_loss_value = min_log_loss_model["model_scores"]['Log Loss']
            best_models_grid = {model_name: model_data for model_name, model_data in filtered_models_grid.items()
                                if model_data["model_scores"]['Log Loss'] == min_log_loss_value}

    elif eval_model == xgb.XGBClassifier:
        # filter out models with precision<10%
        best_models_grid = {model_name: model_data for model_name, model_data in grid.items()
                            if model_data["model_scores"]['Precision'] > precision_filter}

    # return best models
    return best_models_grid


def generate_model_file_name(best_model_name):
    new_model_file_path = str(best_model_name) + "_" + str(datetime.now().strftime("%Y%m%d_%H%M")) + ".pkl"

    return new_model_file_path


def rename_model_pickle_file(new_model_file_path, best_model_data):
    # retrieve best model path
    src = best_model_data["model_file_path"]

    # rename the original file
    os.rename(src, new_model_file_path)


def upload_to_gcs(new_model_file_path, config):
    # retrieve bucket_name from config file
    bucket_name = config["GCS"]["bucket_name"]

    # take that file and upload into GCS
    wi_credentials = compute_engine.Credentials()
    storage_client = storage.Client(credentials=wi_credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(new_model_file_path)
    blob.upload_from_filename(new_model_file_path)


# EXECUTION FUNCTIONS #

def main(config):
    print(f"Main function - started at: \033[1m{datetime.now()}\033[0m")
    # # check if we are running in a container and set memory limits appropriately
    # if os.path.isfile('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    #     with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as limit:
    #         mem = int(limit.read())
    #         resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    # connect redshift
    cur = connect_redshift(config)

    # load data from redshift and reduce its memory usage
    df = reduce_memory_usage(load_data(cur, config))

    # data processing
    df = pre_training_data_processing(df, config)

    # data split
    X_train, X_test, y_train, y_test = data_split(df)

    # delete unnecessary objects from memory
    del df

    # define model type
    eval_model = xgb.XGBClassifier if config["Model_Parameters"]["model_type"] == "CLASSIFIER" else xgb.XGBRegressor

    # save models in a dictionary
    grid = tune_models_hyperparams(eval_model, X_train, y_train, X_test, y_test, config)

    # print models
    print_grid_results(grid)

    # extract best models
    best_models_grid = get_best_models(eval_model, grid, config)
    if not best_models_grid:
        raise ValueError("\033[1m No model has fitted the data well \033[0m")

    # rename best model pkl file and upload it to gcs bucket
    for best_model_name, best_model_data in best_models_grid.items():
        new_model_file_path = generate_model_file_name(best_model_name)
        rename_model_pickle_file(new_model_file_path, best_model_data)
        upload_to_gcs(new_model_file_path, config)


# RUN #

if __name__ == "__main__":
    # read from config file
    config = configparser.ConfigParser()
    config.read("config.ini")
    main(config)
