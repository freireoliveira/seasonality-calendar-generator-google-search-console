from google.oauth2 import service_account
from scipy import fft, signal as sig
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud import secretmanager
import numpy as np
import pandas as pd
import requests
import pyarrow
import json
pd.options.mode.chained_assignment = None  # default='warn'

def get_json(config):
    credentials = 'insert your GCP service account info here'
    
    return credentials

def get_schema():
    return [
        bigquery.SchemaField('query' , 'STRING'),
        bigquery.SchemaField('class', 'STRING'),
        bigquery.SchemaField('amount_of_days', 'INTEGER'),
        bigquery.SchemaField('amount_of_impressions', 'INTEGER'),
        bigquery.SchemaField('jan', 'STRING'),
        bigquery.SchemaField('feb', 'STRING'),
        bigquery.SchemaField('mar', 'STRING'),
        bigquery.SchemaField('apr', 'STRING'),
        bigquery.SchemaField('may', 'STRING'),
        bigquery.SchemaField('jun', 'STRING'),
        bigquery.SchemaField('jul', 'STRING'),
        bigquery.SchemaField('aug', 'STRING'),
        bigquery.SchemaField('sep', 'STRING'),
        bigquery.SchemaField('oct', 'STRING'),
        bigquery.SchemaField('nov', 'STRING'),
        bigquery.SchemaField('dec', 'STRING')
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

def prepare_df(df):
    df['date'] = df['date'].astype('str')
    df['date'] = df['date'].apply(lambda x: x.replace('-', ''))
    df['date'] = df['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

    content = df.sort_values(['date'], ascending=True).groupby('query')[['date', 'impressions']].agg(lambda x: list(x))
    time = pd.date_range(min(df['date']), max(df['date']), freq = 'D').to_series().sort_values()
    return content, time

def factual_checker(df):
  impressions = df['impressions']
  dates = df.index
  impressions = np.array(impressions).astype('float64').astype('int')
  if len(impressions) != 0:
    normalized_impressions = impressions*1./np.max(impressions)
    effective_dates = list()
    for i in range(len(normalized_impressions)):
      if normalized_impressions[i] < 0.5:
        pass
      else:
        effective_dates.append(dates[i])

    str_effective_dates = list()
    for date in effective_dates:
      string = str(date)
      str_date = string.replace(' 00:00:00', '')
      str_effective_dates.append(str_date)
    
    integers = np.array([int(x.replace('-', '')) for x in str_effective_dates])

    if len(effective_dates) == 1:
      return 'True', str_effective_dates
    elif len(effective_dates) > 1 and np.max(np.diff(integers)) < 7:
      return 'True', str_effective_dates
    else:
      return 'False'
  else:
    return 'False'

def month_converter(all):
    month_name = {
        '1': 'jan',
        '2': 'feb',
        '3': 'mar',
        '4': 'apr',
        '5': 'may',
        '6': 'jun',
        '7': 'jul',
        '8': 'aug',
        '9': 'sep',
        '10': 'oct',
        '11': 'nov',
        '12': 'dec'        
    }
    month_converted = list()
    for mm in all:
        month = f'{month_name[str(mm)]}'
        month_converted.append(month)
    return month_converted

def peak_finder(classified, signal, reseted, fft_output, time, period, signal_order):
    N = len(time)
    time_list = time.apply(lambda x: datetime.strptime(str(x).replace(' 00:00:00', ''), '%Y-%m-%d')).tolist()

    if classified == 'Evergreen':
        all_months = [1,2,3,4,5,6,7,8,9,10,11,12]
        all_months = month_converter(all_months)
    elif classified == 'Factual':
        try:
            max = np.where(signal == np.max(signal))
            max = time_list[max[0][0]]
            all_months = [max.month]
            all_months = month_converter(all_months)
        except IndexError:
            pass
    else:
        try:
            if signal_order == 1:
                first = reseted['index'][0]
                first_filtered_fft_output = np.array([f if i == first else 0 for i, f in enumerate(fft_output)])
                first_filtered_sig = fft.ifft(first_filtered_fft_output)[:N].real.tolist()
                max = np.where(first_filtered_sig == np.max(first_filtered_sig))
                max = time_list[max[0][0]]
                all_dates = list()
                all_dates.append(max)

                minimum = max
                while minimum > np.min(time):
                    minimum = minimum - timedelta(days = period)
                    if minimum > np.min(time):
                        all_dates.append(minimum)
                    else:
                        break

                maximum = max
                while maximum < np.max(time):
                    maximum = maximum + timedelta(days = period)
                    if maximum < np.max(time):
                        all_dates.append(maximum)
                    else:
                        break

                all_dates = pd.DataFrame(all_dates)
                all_months = all_dates[0].apply(lambda x: x.month).sort_values().tolist()
                all_months = set(all_months)
                all_months = month_converter(all_months)
            elif signal_order == 2:
                second = reseted['index'][1]
                second_filtered_fft_output = np.array([f if i == second else 0 for i, f in enumerate(fft_output)])
                second_filtered_sig = fft.ifft(second_filtered_fft_output)[:N].real.tolist()
                max = np.where(second_filtered_sig == np.max(second_filtered_sig))
                max = time_list[max[0][0]]
                all_dates = list()
                all_dates.append(max)

                minimum = max
                while minimum > np.min(time):
                    minimum = minimum - timedelta(days = period)
                    if minimum > np.min(time):
                        all_dates.append(minimum)
                    else:
                        break

                maximum = max
                while maximum < np.max(time):
                    maximum = maximum + timedelta(days = period)
                    if maximum < np.max(time):
                        all_dates.append(maximum)
                    else:
                        break

                all_dates = pd.DataFrame(all_dates)
                all_months = all_dates[0].apply(lambda x: x.month).sort_values().tolist()
                all_months = set(all_months)
                all_months = month_converter(all_months)
            elif signal_order == 3:
                    third = reseted['index'][2]
                    third_filtered_fft_output = np.array([f if i == third else 0 for i, f in enumerate(fft_output)])
                    third_filtered_sig = fft.ifft(third_filtered_fft_output)[:N].real.tolist()
                    max = np.where(third_filtered_sig == np.max(third_filtered_sig))
                    max = time_list[max[0][0]]
                    all_dates = list()
                    all_dates.append(max)

                    minimum = max
                    while minimum > np.min(time):
                        minimum = minimum - timedelta(days = period)
                        if minimum > np.min(time):
                            all_dates.append(minimum)
                        else:
                            break

                    maximum = max
                    while maximum < np.max(time):
                        maximum = maximum + timedelta(days = period)
                        if maximum < np.max(time):
                            all_dates.append(maximum)
                        else:
                            break

                    all_dates = pd.DataFrame(all_dates)
                    all_months = all_dates[0].apply(lambda x: x.month).sort_values().tolist()
                    all_months = set(all_months)
                    all_months = month_converter(all_months)
        except KeyError:
            pass
    return all_months


def classifier(content, time, dias_evergreen):
    classified = pd.DataFrame(content.index)
    classified['class'] = np.zeros(shape=[len(classified),1])
    classified['months'] = np.zeros(shape=[len(classified),1]).astype(object)
    classified['amount_of_days'] = np.zeros(shape=[len(classified),1])
    classified['amount_of_impressions'] = np.zeros(shape=[len(classified),1])

    for i in range(len(content)):
        new = content.iloc[i]
        df = pd.DataFrame(data=new['impressions'], index=new['date'])
        df.columns=['impressions']
        counts_dict = df['impressions'].to_dict()
        signal = time.apply(lambda x: counts_dict[x] if x in counts_dict.keys() else 0).astype('float64')
        normalized_signal = signal*1./np.max(signal)
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

        classified['amount_of_impressions'][i] = sum((df['impressions'].astype('float64').astype('int')))

        if len(peaks) != 0:
            output = pd.DataFrame()
            output['index'] = peaks
            output['freq (1/dia)'] = peak_freq
            output['amplitude'] = normalized_peak_power
            output['period (days)'] = 1 / peak_freq
            output['fft'] = fft_output[peaks]
            output = output.sort_values('amplitude', ascending=False)
            reseted = output.sort_values('amplitude', ascending=False).reset_index()
            first_period = reseted['period (days)'][0]
            if len(reseted['period (days)']) > 1:
                second_period = reseted['period (days)'][1]
                if len(reseted['period (days)']) > 2:
                    third_period = reseted['period (days)'][2]
                else:
                    third_period = 0
            else:
                second_period = 0
                third_period = 0

            classified['amount_of_days'][i] = np.max(df.index) - np.min(df.index)
            classified['amount_of_days'][i] = str(classified['amount_of_days'][i]).replace('days', '').replace(' 00:00:00', '')

            if 365*p < first_period and first_period < 365*q:
                if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                    classified['class'][i] = 'Factual'
                else:
                    classified['class'][i] = 'Anual'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 365, 1)
            elif 180*p < first_period and first_period < 180*q:
                if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                    classified['class'][i] = 'Factual'
                else:
                    classified['class'][i] = 'Semestral'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 180, 1)
            elif 90*p < first_period and first_period < 90*q:
                if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                    classified['class'][i] = 'Factual'
                else:
                    classified['class'][i] = 'Trimestral'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 90, 1)
            elif 60*p < first_period and first_period < 60*q:
                if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                    classified['class'][i] = 'Factual'
                else:
                    if first_period <= dias_evergreen:
                        classified['class'][i] = 'Evergreen'
                    else:
                        classified['class'][i] = 'Bimestral'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 60, 1)
            elif 30*p < first_period and first_period < 30*q:
                if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                    classified['class'][i] = 'Factual'
                else:
                    if first_period <= dias_evergreen:
                        classified['class'][i] = 'Evergreen'
                    else:
                        classified['class'][i] = 'Mensal'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 30, 1)
            elif 15*p < first_period and first_period < 15*q:
                if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                    classified['class'][i] = 'Factual'
                else:
                    if first_period <= dias_evergreen:
                        classified['class'][i] = 'Evergreen'
                    else:
                        classified['class'][i] = 'Quinzenal'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 15, 1)
            elif 7*p < first_period and first_period < 7*q:
                if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                    classified['class'][i] = 'Factual'
                else:
                    if first_period <= dias_evergreen:
                        classified['class'][i] = 'Evergreen'
                    else:
                        classified['class'][i] = 'Semanal'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 7, 1)
            elif 1*p < first_period and first_period < 1*q:
                if first_period <= dias_evergreen:
                    classified['class'][i] = 'Evergreen'
                else:
                    classified['class'][i] = 'Diário'
                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 1, 1)
            else:
                if 365*p < second_period and second_period < 365*q:
                    if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                        classified['class'][i] = 'Factual'
                    else:
                        classified['class'][i] = 'Anual'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 365, 2)
                elif 180*p < second_period and second_period < 180*q:
                    if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                        classified['class'][i] = 'Factual'
                    else:
                        classified['class'][i] = 'Semestral'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 180, 2)
                elif 90*p < second_period and second_period < 90*q:
                    if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                        classified['class'][i] = 'Factual'
                    else:
                        classified['class'][i] = 'Trimestral'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 90, 2)
                elif 60*p < second_period and second_period < 60*q:
                    if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                        classified['class'][i] = 'Factual'
                    else:
                        if second_period <= dias_evergreen:
                            classified['class'][i] = 'Evergreen'
                        else:
                            classified['class'][i] = 'Bimestral'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 60, 2)
                elif 30*p < second_period and second_period < 30*q:
                    if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                        classified['class'][i] = 'Factual'
                    else:
                        if second_period <= dias_evergreen:
                            classified['class'][i] = 'Evergreen'
                        else:
                            classified['class'][i] = 'Mensal'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 30, 2)
                elif 15*p < second_period and second_period < 15*q:
                    if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                        classified['class'][i] = 'Factual'
                    else:
                        if second_period <= dias_evergreen:
                            classified['class'][i] = 'Evergreen'
                        else:
                            classified['class'][i] = 'Quinzenal'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 15, 2)
                elif 7*p < second_period and second_period < 7*q:
                    if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                        classified['class'][i] = 'Factual'
                    else:
                        if second_period <= dias_evergreen:
                            classified['class'][i] = 'Evergreen'
                        else:
                            classified['class'][i] = 'Semanal'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 7, 2)
                elif 1*p < second_period and second_period < 1*q:
                    if second_period <= dias_evergreen:
                        classified['class'][i] = 'Evergreen'
                    else:
                        classified['class'][i] = 'Diário'
                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 1, 2)
                else:
                    if 365*p < third_period and third_period < 365*q:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                        else:
                            classified['class'][i] = 'Anual'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 365, 3)
                    elif 180*p < third_period and third_period < 180*q:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                        else:
                            classified['class'][i] = 'Semestral'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 180, 3)
                    elif 90*p < third_period and third_period < 90*q:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                        else:
                            classified['class'][i] = 'Trimestral'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 90, 3)
                    elif 60*p < third_period and third_period < 60*q:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                        else:
                            if third_period <= dias_evergreen:
                                classified['class'][i] = 'Evergreen'
                            else:
                                classified['class'][i] = 'Bimestral'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 60, 3)
                    elif 30*p < third_period and third_period < 30*q:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                        else:
                            if third_period <= dias_evergreen:
                                classified['class'][i] = 'Evergreen'
                            else:
                                classified['class'][i] = 'Mensal'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 30, 3)
                    elif 15*p < third_period and third_period < 15*q:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                        else:
                            if third_period <= dias_evergreen:
                                classified['class'][i] = 'Evergreen'
                            else:
                                classified['class'][i] = 'Quinzenal'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 15, 3)
                    elif 7*p < third_period and third_period < 7*q:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                        else:
                            if third_period <= dias_evergreen:
                                classified['class'][i] = 'Evergreen'
                            else:
                                classified['class'][i] = 'Semanal'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 7, 3)
                    elif 1*p < third_period and third_period < 1*q:
                        if third_period <= dias_evergreen:
                            classified['class'][i] = 'Evergreen'
                        else:
                            classified['class'][i] = 'Diário'
                        classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 1, 3)
                    else:
                        if factual_checker(df)[0] == 'True' or sum(normalized_signal[-10:]) < 0.2:
                            classified['class'][i] = 'Factual'
                            classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 1, 1)
                        else:
                            season = int(reseted['period (days)'][0])
                            if season > 365*q:
                                classified['class'][i] = 'Padrão não detectado'
                                classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 1, 1)
                            else:
                                if season <= dias_evergreen and sum(normalized_signal[-10:]) > 0.2:
                                    classified['class'][i] = 'Evergreen'
                                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 1, 1)
                                else:
                                    classified['class'][i] = f'Padrão se repete a cada {season} dias'
                                    classified['months'][i] = peak_finder(classified['class'][i], signal, reseted, fft_output, time, 1, 1)
        else:
            classified['class'][i] = 'Padrão não detectado'
    classified['class'] = classified['class'].replace(0.0, 'Padrão não detectado').astype('str')
    classified['query'] = classified['query'].astype('str')
    classified['amount_of_impressions'] = classified['amount_of_impressions'].astype('int')
    classified['amount_of_days'] = classified['amount_of_days'].astype('int')
    classified = classified.explode('months')
    classified['marks'] = np.full(shape=[len(classified),1], fill_value=1)
    calendar = pd.pivot_table(classified, values='marks', index='query', columns='months').fillna(0)
    calendar = calendar.replace(1.0, 'OK')
    calendar = calendar.replace(0.0, '-')
    classified = classified.merge(calendar, how='left', on='query')
    classified = classified.drop(columns=['months', 'marks', 0.0]).drop_duplicates().reset_index()
    cols = ['query', 'class', 'amount_of_days', 'amount_of_impressions', 'jan', 'feb', 
    'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    classified = classified[cols]

    months = classified.filter(classified.columns[-12:])
    for i in range(len(classified)):
        check_ok = list(set(months.iloc[i]))
        if len(check_ok) == 1 and check_ok[0] == 'OK':
            classified['class'][i] = 'Evergreen'

    return classified

def to_bq(classified, json):
    # establish a BigQuery client
    client = bigquery.Client.from_service_account_info(json)
    classified = pd.DataFrame(classified)
    dataset_id = 'dataset_id'
    table_name = 'table_name'
    job_config = bigquery.LoadJobConfig()
    table_ref = client.dataset(dataset_id).table(table_name)
    job_config.schema = get_schema()
    job_config.write_disposition = 'WRITE_TRUNCATE'

    load_job = client.load_table_from_dataframe(classified, table_ref, job_config=job_config)
    load_job.result()

def main():

    config = {
        "project_id": "project_id",
        "dataset_id": "dataset_id"
    }
    
    json = get_json(config)
    credentials = service_account.Credentials.from_service_account_info(json)
    dias_evergreen = 40
    df = get_data_from_bq(credentials)
    content, time = prepare_df(df)
    classified = classifier(content, time, dias_evergreen)
    to_bq(classified, json)

main()
