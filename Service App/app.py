import ast
import configparser
import psycopg2
from flask import Flask, make_response, request, render_template, Response
import io
from io import StringIO
import pandas as pd
from model_utilities import create_predictions_df


# initialize flask app
app = Flask(__name__)


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


def split_channels(df):
    # calculate no of buckets to split
    n_k, n_m = divmod(len(df), 1000)
    n_m = 1 if n_m > 0 else 0
    n = 1 if len(df) <= 1000 else n_k + n_m

    # calculate buckets size
    k, m = divmod(len(df), n)

    return (df[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def input_data_processing(df, cur):
    # list to store features
    result = []

    # retrieve channels from input df (curation list)
    channels_list = df['channel_id'].values.tolist()

    # add features for each channel
    for channels_bucket in split_channels(channels_list):
        query = f"""select *                       
                    from dev_yehuda.acceptance_rate_model_creators_current_features
                    where channel_id in ({",".join([f"'{channel}'" for channel in channels_bucket])})
                    """
        cur.execute(query)
        df_query = pd.DataFrame(cur.fetchall())
        df_query.columns = [desc[0] for desc in cur.description]
        result.append(df_query)

    # join channels and features
    df_features = pd.concat(result)
    df = df.merge(df_features, on='channel_id')

    return df


def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


# create html page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# health check route
@app.route("/health")
def health_check():
    print("/health request")
    status_code = Response(status=200)
    return status_code


# create flask app
@app.route('/predict', methods=["POST"])
def predict():
    # request csv file
    f = request.files['data_file']
    if not f:
        return "No file"

    # read data from csv file
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())

    # convert the data into df
    df = pd.read_csv(StringIO(result))

    # get variables from config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # connect redshift
    cur = connect_redshift(config)

    # add features to csv file
    df = input_data_processing(df, cur)

    # predict
    df = create_predictions_df(df)

    # return predictions
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0")
