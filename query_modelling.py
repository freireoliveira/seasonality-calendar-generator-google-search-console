from google.oauth2 import service_account
from scipy import fft, signal as sig
from datetime import datetime
from google.cloud import bigquery
from google.cloud import secretmanager
import numpy as np
import pandas as pd
import requests
import pyarrow
import json
pd.options.mode.chained_assignment = None  # default='warn'

def get_json():
    credentials = 'insert your GCP service account info here'
    
    return credentials

def get_schema():
    return [
        bigquery.SchemaField('query' , 'STRING'),
        bigquery.SchemaField('signal_0', 'FLOAT'),
        bigquery.SchemaField('signal_1', 'FLOAT'),
        bigquery.SchemaField('signal_2', 'FLOAT'),
        bigquery.SchemaField('signal_3', 'FLOAT'),
        bigquery.SchemaField('date', 'DATE')
    ]

def get_data_from_bq(credentials):
    bqclient = bigquery.Client(credentials=credentials)

    # Download query results.
    query_string = """
    SELECT
        query,
        SUM(impressions) AS impressions,
        date
    FROM
        `table`
    GROUP BY
        query,
        date
    ORDER BY
        date ASC,
        impressions DESC
    """

    dataframe = (
        bqclient.query(query_string)
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
        )
    )
    return dataframe

def check_pattern(df, credentials):
    bqclient = bigquery.Client(credentials=credentials)

    # Download query results.
    query_string = """
    SELECT
        DISTINCT query
    FROM
        `table_generated_by_query_classifier.py`
    WHERE
        class = 'Padrão não detectado'
    """

    dataframe = (
        bqclient.query(query_string)
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
        )
    )
    
    remove = dataframe['query'].tolist()
    df = df[~df['query'].isin(remove)].reset_index()

    return df

def prepare_df(df):
    df['date'] = df['date'].astype('str')
    df['date'] = df['date'].apply(lambda x: x.replace('-', ''))
    df['date'] = df['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

    content = df.sort_values(['date'], ascending=True).groupby('query')[['date', 'impressions']].agg(lambda x: list(x))
    time = pd.date_range(min(df['date']), max(df['date']), freq = 'D').to_series().sort_values()
    return df, content, time

def classifier(content, time):
    modeled = pd.DataFrame(content.index)
    modeled['signal_0'] = np.zeros(shape=[len(modeled),1]).astype(object)
    modeled['signal_1'] = np.zeros(shape=[len(modeled),1]).astype(object)
    modeled['signal_2'] = np.zeros(shape=[len(modeled),1]).astype(object)
    modeled['signal_3'] = np.zeros(shape=[len(modeled),1]).astype(object)
    modeled['date'] = np.zeros(shape=[len(modeled),1]).astype(object)

    for i in range(len(content)):
        new = content.iloc[i]
        df = pd.DataFrame(data=new['impressions'], index=new['date'])
        df.columns=['impressions']
        counts_dict = df['impressions'].to_dict()
        signal = time.apply(lambda x: counts_dict[x] if x in counts_dict.keys() else 0).astype('float64')
        signal = signal - signal.mean()         #signal as mean deviation

        fft_output = fft.fft(np.array(signal))
        power = np.abs(fft_output)       #absolute amplitude (no imaginary part)
        freq = fft.fftfreq(len(signal))        #fft frequencies

        mask = freq >= 0          #only positive or null frequencies
        freq = freq[mask]
        power = power[mask]

        p = 0.85
        q = 1-p+1
        peaks = sig.find_peaks(power[freq >=0], prominence=10**2)[0]
        peak_freq =  freq[peaks]
        peak_power = power[peaks]
        period = 1 / peak_freq
        if len(peak_power) > 1:
            first = period[0]
            normalized_peak_power = peak_power*1./np.max(peak_power)
            if first > 365*q and normalized_peak_power[1] > 0.2:
                peaks = peaks[1:]
                period = period[1:]
                peak_freq = peak_freq[1:]
                peak_power = peak_power[1:]
                normalized_peak_power = peak_power*1./np.max(peak_power)
            else:
                pass
        else:
            normalized_peak_power = 1

        output = pd.DataFrame()
        output['index'] = peaks
        output['amplitude'] = normalized_peak_power
        output['period (days)'] = period
        output = output.sort_values('amplitude', ascending=False)
        reseted = output.sort_values('amplitude', ascending=False).reset_index()

        N = len(time)
        time_list = time.apply(lambda x: datetime.strptime(str(x).replace(' 00:00:00', ''), '%Y-%m-%d')).tolist()

        try:
            first = reseted['index'][0]
            first_filtered_fft_output = np.array([f if i == first else 0 for i, f in enumerate(fft_output)])
            first_filtered_sig = fft.ifft(first_filtered_fft_output)[:N].real.tolist()
            modeled['date'][i] = time_list
            modeled['signal_0'][i] = signal.tolist()
            modeled['signal_1'][i] = first_filtered_sig

            if len(peak_freq) > 1:
                second = reseted['index'][1]
                second_filtered_fft_output = np.array([f if i == second else 0 for i, f in enumerate(fft_output)])
                second_filtered_sig = fft.ifft(second_filtered_fft_output)[:N].real.tolist()
                modeled['signal_2'][i] = second_filtered_sig
                if len(peak_freq) > 2:
                    third = reseted['index'][2]
                    third_filtered_fft_output = np.array([f if i == third else 0 for i, f in enumerate(fft_output)])
                    third_filtered_sig = fft.ifft(third_filtered_fft_output)[:N].real.tolist()
                    modeled['signal_3'][i] = third_filtered_sig
                else:
                    third_filtered_sig = np.zeros(shape=len(fft_output))
                    modeled['signal_3'][i] = third_filtered_sig
            else:
                second_filtered_sig = np.zeros(shape=len(fft_output))
                third_filtered_sig = np.zeros(shape=len(fft_output))
                modeled['signal_2'][i] = second_filtered_sig
                modeled['signal_3'][i] = third_filtered_sig
        except IndexError:
            pass

    final = modeled.explode('signal_0')
    final['signal_0'] = final['signal_0'] / 5
    other_signal = modeled.explode('signal_1')
    final = final.drop('signal_1', axis=1).assign(signal_1=other_signal['signal_1'])
    other_signal = modeled.explode('signal_2')
    final = final.drop('signal_2', axis=1).assign(signal_2=other_signal['signal_2'])
    other_signal = modeled.explode('signal_3')
    final = final.drop('signal_3', axis=1).assign(signal_3=other_signal['signal_3'])
    other_signal = modeled.explode('date')
    final = final.drop('date', axis=1).assign(date=other_signal['date'])
    
    return final

def to_bq(modeled, json):
    # establish a BigQuery client
    client = bigquery.Client.from_service_account_json(json)
    modeled = pd.DataFrame(modeled)
    dataset_id = 'dataset_id'
    table_name = 'table_name_modelling'
    job_config = bigquery.LoadJobConfig()
    table_ref = client.dataset(dataset_id).table(table_name)
    job_config.schema = get_schema()
    job_config.write_disposition = 'WRITE_TRUNCATE'

    load_job = client.load_table_from_dataframe(modeled, table_ref, job_config=job_config)
    load_job.result()

def main():
    json = get_json()
    credentials = service_account.Credentials.from_service_account_info(json)
    df = get_data_from_bq(credentials)
    df = check_pattern(df, credentials)
    df, content, time = prepare_df(df)
    modeled = classifier(content, time)
    #to_bq(modeled)

main()
