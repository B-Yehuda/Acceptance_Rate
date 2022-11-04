import psycopg2
from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
import os
from model_utilities import create_predictions_df


# initialize flask app
app = Flask(__name__)


def credentials():
    # redshift credentials
    credentials = {'host':,
                   'port':,
                   'dbname':,
                   'user': ,
                   'password':}

    return credentials


def connect_redshift():
    # get credentials
    redshift_credentials = credentials()

    # connect redshift
    connection = psycopg2.connect(
        host=redshift_credentials['host'],
        port=redshift_credentials['port'],
        dbname=redshift_credentials['dbname'],
        user=redshift_credentials['user'],
        password=redshift_credentials['password'])

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


def add_features_to_df(df):
    # list to store features
    result = []

    # connect redshift
    cur = connect_redshift()

    # retrieve channels from input df (curation list)
    channels_list = df['channel_id'].values.tolist()

    # add features for each channel
    for channels_bucket in split_channels(channels_list):
        query = f"""select channel_id
                         , se_age_days::float
                         , country
                         , language
                         , ccv_30_d::float
                         , ccv_60_d::float
                         , ccv_growth_60_30_d::float
                         , most_played_game
                         , cnt_streams::float
                         , invitations_l3m::float
                         , acceptances_l3m::float
                         , deployments_l3m::float
                         , rejections_l3m::float
                         , offer_page_visits_l3m::float
                         , invitations_l6m::float
                         , acceptances_l6m::float
                         , deployments_l6m::float
                         , rejections_l6m::float
                         , offer_page_visits_l6m::float
                         , hours_streamed::float
                         , hours_watched::float
                         , total_chatters::float
                         , is_tipping_panel
                         , is_bot_command_usage
                         , is_overlay
                         , is_website_visit
                         , is_se_live
                         , is_alert_box_fired
                         , cnt_alert_box_fired::float
                         , is_open_stream_report
                         , campaigns_revenue::float
                         , tips::float
                         , tips_revenue::float
                         , on_screen_cheers::float
                         , on_screen_cheers_revenue::float
                         , on_screen_subs::float
                         , on_screen_subs_revenue::float
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


# create flask app
@app.route('/csv_predictions', methods=["POST"])
def transform_view():
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

    # add features to csv file
    df = add_features_to_df(df)

    # predict
    df = create_predictions_df(df)

    # return predictions
    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


if __name__ == "__main__":
    app.run(debug=True)
