####################################################################################################
##                                   LIBRARIES AND DEPENDENCIES                                   ##
####################################################################################################

# Tethys platform
from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from tethys_sdk.routing import controller
from tethys_sdk.gizmos import PlotlyView, DatePicker
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes

# Postgresql
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from pandas_geojson import to_geojson

# Geoglows
import math
import geoglows
import requests
import numpy as np
import datetime as dt
import HydroErr as he
import hydrostats as hs
from scipy import stats
import hydrostats.data as hd
import plotly.graph_objs as go


# Base
import io
import os
from dotenv import load_dotenv


####################################################################################################
##                                       STATUS VARIABLES                                         ##
####################################################################################################

# Import enviromental variables 
load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME = os.getenv('DB_NAME')

# Generate the conection token
global tokencon
tokencon = "postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER, DB_PASS, DB_NAME)



####################################################################################################
##                                 UTILS AND AUXILIAR FUNCTIONS                                   ##
####################################################################################################

def get_format_data(sql_statement, conn):
    # Retrieve data from database
    data =  pd.read_sql(sql_statement, conn)
    # Datetime column as dataframe index
    data.index = data.datetime
    data = data.drop(columns=['datetime'])
    # Format the index values
    data.index = pd.to_datetime(data.index)
    data.index = data.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    data.index = pd.to_datetime(data.index)
    # Return result
    return(data)


def get_bias_corrected_data(sim, obs):

    # Fix negative values
    if sim.min().values[0] < 0:
        sim = sim + abs(sim.min().values[0])
    if obs.min().values[0] < 0:
        obs = obs + abs(obs.min().values[0])

    outdf = geoglows.bias.correct_historical(sim, obs)
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)


def gumbel_1(std: float, xbar: float, rp: int or float) -> float:
  return -math.log(-math.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)


def get_return_periods(comid, data):
    # Stats
    max_annual_flow = data.groupby(data.index.strftime("%Y")).max()
    mean_value = np.mean(max_annual_flow.iloc[:,0].values)
    std_value = np.std(max_annual_flow.iloc[:,0].values)
    # Return periods
    return_periods = [100, 50, 25, 10, 5, 2]
    return_periods_values = []
    # Compute the corrected return periods
    for rp in return_periods:
      return_periods_values.append(gumbel_1(std_value, mean_value, rp))
    # Parse to list
    d = {'rivid': [comid], 
         'return_period_100': [return_periods_values[0]], 
         'return_period_50': [return_periods_values[1]], 
         'return_period_25': [return_periods_values[2]], 
         'return_period_10': [return_periods_values[3]], 
         'return_period_5': [return_periods_values[4]], 
         'return_period_2': [return_periods_values[5]]}
    # Parse to dataframe
    corrected_rperiods_df = pd.DataFrame(data=d)
    corrected_rperiods_df.set_index('rivid', inplace=True)
    return(corrected_rperiods_df)


# Calc 7q10 threshold for high warnings levels
def get_warning_low_level(comid, data):

    def __calc_method__(ts):
        # Result dictionary
        rv = {'empirical' : {},
                'norm'      : {'fun'  : stats.norm,
                                'para' : {'loc'   : np.nanmean(ts), 
                                        'scale' : np.nanstd(ts)}},
                'pearson3'  : {'fun' : stats.pearson3,
                                'para' : {'loc'   : np.nanmean(ts), 
                                        'scale' : np.nanstd(ts), 
                                        'skew'  : 1}},
                'dweibull'  : {'fun' : stats.dweibull,
                                'para' : {'loc'   : np.nanmean(ts), 
                                        'scale' : np.nanstd(ts), 
                                        'c'     : 1}},
                'chi2'      : {'fun' : stats.chi2,
                                'para' : {'loc'   : np.nanmean(ts), 
                                        'scale' : np.nanstd(ts), 
                                        'df'    : 2}},
                'gumbel_r'  : {'fun' : stats.gumbel_r,
                                'para' : {'loc'   : np.nanmean(ts) - 0.45005 * np.nanstd(ts),
                                        'scale' : 0.7797 * np.nanstd(ts)}}}

        # Extract empirical distribution data
        freq, cl = np.histogram(ts, bins='sturges')
        freq = np.cumsum(freq) / np.sum(freq)
        cl_marc = (cl[1:] + cl[:-1]) / 2

        # Save values
        rv['empirical'].update({'freq'    : freq,
                                'cl_marc' : cl_marc})

        # Function for stadistical test
        ba_xi2 = lambda o, e : np.square(np.subtract(o,e)).mean() ** (1/2)

        # Add to probability distribution the cdf and the xi test
        for p_dist in rv:
            if p_dist == 'empirical':
                continue
            
            # Build cummulative distribution function (CDF)
            rv[p_dist].update({'cdf' : rv[p_dist]['fun'].cdf(x = cl_marc, 
                                                                **rv[p_dist]['para'])})
            
            # Obtain the xi test result
            rv[p_dist].update({f'{p_dist}_x2test' : ba_xi2(o = rv[p_dist]['cdf'], 
                                                            e = freq)})
        
        # Select best probability function
        p_dist_comp = pd.DataFrame(data={'Distribution' : [p_dist for p_dist in rv if p_dist != 'empirical'],
                                         'xi2_test'     : [rv[p_dist][f'{p_dist}_x2test'] for p_dist in rv if p_dist != 'empirical']})
        p_dist_comp.sort_values(by='xi2_test', inplace = True)
        p_dist_comp.reset_index(drop = True, inplace = True)
        best_p_dist = p_dist_comp['Distribution'].values[0]
        
        # NOTES:
        # 
        # Q -> Prob
        # rv[best_p_dist]['fun'](**rv[best_p_dist]['para']).pdf()
        #
        # Q -> Prob acum
        # rv[best_p_dist]['fun'](**rv[best_p_dist]['para']).cdf()
        #
        # Prob acum -> Q
        # rv[best_p_dist]['fun'](**rv[best_p_dist]['para']).ppf([0.15848846])

        return rv[best_p_dist]['fun'](**rv[best_p_dist]['para'])
    
    # Previous datatime manager
    data_cp = data.copy()
    data_cp = data_cp.rolling(window=7).mean()
    data_cp = data_cp.groupby(data_cp.index.year).min().values.flatten()

    # Calc comparation value
    rv = {}
    for key in {'7q10' : 1}:
        res = __calc_method__(data_cp)
        # TODO: Fix in case of small rivers get 7q10 negative
        val = res.ppf([1/10]) if res.ppf([1/10]) > 0 else 0
        rv.update({key : val})


    # Build result dataframe
    d = {'rivid': [comid]}
    d.update(rv)

    # Parse to dataframe
    corrected_low_warnings_df = pd.DataFrame(data=d)
    corrected_low_warnings_df.set_index('rivid', inplace=True)

    return corrected_low_warnings_df


def ensemble_quantile(ensemble, quantile, label):
    df = ensemble.quantile(quantile, axis=1).to_frame()
    df.rename(columns = {quantile: label}, inplace = True)
    return(df)


def ensemble_median(ensemble):
    df = ensemble.median(axis=1).to_frame()
    df.rename(columns = {0: 'flow_median_m^3/s'}, inplace = True)
    return df



def get_ensemble_stats(ensemble):
    high_res_df = ensemble['ensemble_52_m^3/s'].to_frame()
    ensemble.drop(columns=['ensemble_52_m^3/s'], inplace=True)
    ensemble.dropna(inplace= True)
    high_res_df.dropna(inplace= True)
    high_res_df.rename(columns = {'ensemble_52_m^3/s':'high_res_m^3/s'}, inplace = True)

    stats_df = pd.concat([
        ensemble_quantile(ensemble, 1.00, 'flow_max_m^3/s'),
        ensemble_quantile(ensemble, 0.75, 'flow_75%_m^3/s'),
        ensemble_quantile(ensemble, 0.50, 'flow_avg_m^3/s'),
        ensemble_quantile(ensemble, 0.25, 'flow_25%_m^3/s'),
        ensemble_quantile(ensemble, 0.00, 'flow_min_m^3/s'),
        ensemble_median(ensemble),
        high_res_df
    ], axis=1)
    return(stats_df)


def __bias_correction_forecast__(sim_hist, fore_nofix, obs_hist):
    '''Correct Bias Forecasts'''

    # Selection of monthly simulated data
    monthly_simulated = sim_hist[sim_hist.index.month == (fore_nofix.index[0]).month].dropna()

    # Obtain Min and max value
    min_simulated = monthly_simulated.min().values[0]
    max_simulated = monthly_simulated.max().values[0]

    min_factor_df   = fore_nofix.copy()
    max_factor_df   = fore_nofix.copy()
    forecast_ens_df = fore_nofix.copy()

    for column in fore_nofix.columns:
        # Min Factor
        tmp_array = np.ones(fore_nofix[column].shape[0])
        tmp_array[fore_nofix[column] < min_simulated] = 0
        min_factor = np.where(tmp_array == 0, fore_nofix[column] / min_simulated, tmp_array)

        # Max factor
        tmp_array = np.ones(fore_nofix[column].shape[0])
        tmp_array[fore_nofix[column] > max_simulated] = 0
        max_factor = np.where(tmp_array == 0, fore_nofix[column] / max_simulated, tmp_array)

        # Replace
        tmp_fore_nofix = fore_nofix[column].copy()
        tmp_fore_nofix.mask(tmp_fore_nofix <= min_simulated, min_simulated, inplace=True)
        tmp_fore_nofix.mask(tmp_fore_nofix >= max_simulated, max_simulated, inplace=True)

        # Save data
        forecast_ens_df.update(pd.DataFrame(tmp_fore_nofix, index=fore_nofix.index, columns=[column]))
        min_factor_df.update(pd.DataFrame(min_factor, index=fore_nofix.index, columns=[column]))
        max_factor_df.update(pd.DataFrame(max_factor, index=fore_nofix.index, columns=[column]))

    # Get  Bias Correction
    corrected_ensembles = geoglows.bias.correct_forecast(forecast_ens_df, sim_hist, obs_hist)
    corrected_ensembles = corrected_ensembles.multiply(min_factor_df, axis=0)
    corrected_ensembles = corrected_ensembles.multiply(max_factor_df, axis=0)

    return corrected_ensembles



def get_corrected_forecast_records(records_df, simulated_df, observed_df):
    ''' Este es el comentario de la doc '''
    date_ini = records_df.index[0]
    month_ini = date_ini.month
    date_end = records_df.index[-1]
    month_end = date_end.month
    meses = np.arange(month_ini, month_end + 1, 1)
    fixed_records = pd.DataFrame()
    for mes in meses:
        values = records_df.loc[records_df.index.month == mes]
        monthly_simulated = simulated_df[simulated_df.index.month == mes].dropna()
        monthly_observed = observed_df[observed_df.index.month == mes].dropna()
        min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
        max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())
        min_factor_records_df = values.copy()
        max_factor_records_df = values.copy()
        fixed_records_df = values.copy()
        column_records = values.columns[0]
        tmp = records_df[column_records].dropna().to_frame()
        min_factor = tmp.copy()
        max_factor = tmp.copy()
        min_factor.loc[min_factor[column_records] >= min_simulated, column_records] = 1
        min_index_value = min_factor[min_factor[column_records] != 1].index.tolist()
        for element in min_index_value:
            min_factor[column_records].loc[min_factor.index == element] = tmp[column_records].loc[tmp.index == element] / min_simulated
        max_factor.loc[max_factor[column_records] <= max_simulated, column_records] = 1
        max_index_value = max_factor[max_factor[column_records] != 1].index.tolist()
        for element in max_index_value:
            max_factor[column_records].loc[max_factor.index == element] = tmp[column_records].loc[tmp.index == element] / max_simulated
        tmp.loc[tmp[column_records] <= min_simulated, column_records] = min_simulated
        tmp.loc[tmp[column_records] >= max_simulated, column_records] = max_simulated
        fixed_records_df.update(pd.DataFrame(tmp[column_records].values, index=tmp.index, columns=[column_records]))
        min_factor_records_df.update(pd.DataFrame(min_factor[column_records].values, index=min_factor.index, columns=[column_records]))
        max_factor_records_df.update(pd.DataFrame(max_factor[column_records].values, index=max_factor.index, columns=[column_records]))
        corrected_values = geoglows.bias.correct_forecast(fixed_records_df, simulated_df, observed_df)
        corrected_values = corrected_values.multiply(min_factor_records_df, axis=0)
        corrected_values = corrected_values.multiply(max_factor_records_df, axis=0)
        fixed_records = fixed_records.append(corrected_values)
    fixed_records.sort_index(inplace=True)
    return(fixed_records)


def get_forecast_date(comid, date):
    url = 'https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id={0}&date={1}&return_format=csv'.format(comid, date)
    status = False
    while not status:
      try:
        outdf = pd.read_csv(url, index_col=0)
        status = True
      except:
        print("Trying to retrieve data...")
    
    # Filter and correct data
    outdf[outdf < 0] = 0
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)


def get_forecast_record_date(comid, date):
    idate = dt.datetime.strptime(date, '%Y%m%d') - dt.timedelta(days=10)
    idate = idate.strftime('%Y%m%d')
    url = 'https://geoglows.ecmwf.int/api/ForecastRecords/?reach_id={0}&start_date={1}&end_date={2}&return_format=csv'.format(comid, idate, date)
    status = False
    while not status:
      try:
        outdf = pd.read_csv(url, index_col=0)
        status = True
      except:
        print("Trying to retrieve data...")
    # Filter and correct data
    outdf[outdf < 0] = 0
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)


def get_fews_data(station_code):
    # TODO : Change with fews postgres implementation
    url = 'http://fews.ideam.gov.co/colombia/jsonH/00' + station_code + 'Hobs.json'

    try:
        # Call data
        f = requests.get(url, verify=False)
        data = f.json()

        # Extract data
        observedDischarge = (data.get('obs'))
        sensorDischarge = (data.get('sen'))
        observedDischarge = (observedDischarge.get('data'))
        sensorDischarge = (sensorDischarge.get('data'))
        datesObservedDischarge = [row[0] for row in observedDischarge]
        observedDischarge = [row[1] for row in observedDischarge]
        datesSensorDischarge = [row[0] for row in sensorDischarge]
        sensorDischarge = [row[1] for row in sensorDischarge]

        # Build dataframe discharge
        observedDischarge_df = pd.DataFrame(data={'date' : datesObservedDischarge,
                                                  'water level cm' : observedDischarge})
        observedDischarge_df['date'] = pd.to_datetime(observedDischarge_df['date'], format='%Y/%m/%d %H:%M')
        observedDischarge_df['water level cm'] = observedDischarge_df['water level cm'].astype(float) * 100
        observedDischarge_df.dropna(inplace=True)
        observedDischarge_df.set_index('date', inplace = True)

        # Build dataframe sensor
        sensorDischarge_df   = pd.DataFrame(data={'date' : datesSensorDischarge,
                                                  'water level cm' : sensorDischarge})
        sensorDischarge_df['date'] = pd.to_datetime(sensorDischarge_df['date'], format='%Y/%m/%d %H:%M')
        sensorDischarge_df['water level cm'] = sensorDischarge_df['water level cm'].astype(float) * 100
        sensorDischarge_df.dropna(inplace=True)
        sensorDischarge_df.set_index('date', inplace=True)

    except Exception as e:

        # Build discharge dataframe
        observedDischarge_df = pd.DataFrame(data = {'date' : [pd.NaT],
                                                    'water level cm' : [np.nan]})
        observedDischarge_df.set_index('date', inplace = True)

        # Build sensor dataframe
        sensorDischarge_df = pd.DataFrame(data = {'date' : [pd.NaT],
                                                  'water level cm' : [np.nan]})
        sensorDischarge_df.set_index('date', inplace=True)

    return observedDischarge_df, sensorDischarge_df


####################################################################################################
##                                      PLOTTING FUNCTIONS                                        ##
####################################################################################################

# Plotting daily averages values
def get_daily_average_plot(merged_sim, merged_cor, code, name):
    # Generate the average values
    daily_avg_sim = hd.daily_average(merged_sim)
    daily_avg_cor = hd.daily_average(merged_cor)
    # Generate the plots on Ploty
    daily_avg_obs_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 1].values, name = 'Observado', line=dict(color="#636EFA"))
    daily_avg_corr_sim_Q = go.Scatter(x = daily_avg_cor.index, y = daily_avg_cor.iloc[:, 0].values, name = 'Simulado corregida', line=dict(color="#00CC96"))
    # PLot Layout
    layout = go.Layout(
        title='Nivel promedio diario <br> {0} - {1}'.format(code.upper(), name),
        xaxis=dict(title='Dia', ), 
        yaxis=dict(title='Nivel de agua (cm)', autorange=True),
        showlegend=True)
    # Generate the output
    chart_obj = go.Figure(data=[daily_avg_obs_Q, daily_avg_corr_sim_Q], layout=layout)
    return(chart_obj)



# Plotting monthly averages values
def get_monthly_average_plot(merged_sim, merged_cor, code, name):
    # Generate the average values
    daily_avg_sim = hd.monthly_average(merged_sim)
    daily_avg_cor = hd.monthly_average(merged_cor)
    # Generate the plots on Ploty
    daily_avg_obs_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 1].values, name = 'Observed', line=dict(color="#636EFA"))
    daily_avg_corr_sim_Q = go.Scatter(x = daily_avg_cor.index, y = daily_avg_cor.iloc[:, 0].values, name = 'Corrected Simulated', line=dict(color="#00CC96"))
    # PLot Layout
    layout = go.Layout(
        title='Nivel promedio mensual <br> {0} - {1}'.format(code.upper(), name),
        xaxis=dict(title='Mes', ), 
        yaxis=dict(title='Nivel de agua (cm)', autorange=True),
        showlegend=True)
    # Generate the output
    chart_obj = go.Figure(data=[daily_avg_obs_Q, daily_avg_corr_sim_Q], layout=layout)
    return(chart_obj)



# Scatter plot (Simulated/Corrected vs Observed)
def get_scatter_plot(merged_sim, merged_cor, code, name, log):
    # Generate Scatter (cor vs obs)
    scatter_data2 = go.Scatter(
        x = merged_cor.iloc[:, 0].values,
        y = merged_cor.iloc[:, 1].values,
        mode='markers',
        name='Corregido',
        marker=dict(color='#00cc96'))
    # Get the max and min values
    min_value = min(min(merged_cor.iloc[:, 1].values), min(merged_cor.iloc[:, 0].values))
    max_value = max(max(merged_cor.iloc[:, 1].values), max(merged_cor.iloc[:, 0].values))
    # Construct the line 1:1
    line_45 = go.Scatter(
        x = [min_value, max_value],
        y = [min_value, max_value],
        mode = 'lines',
        name = 'Línea de 45<suo>o</sup>',
        line = dict(color='black'))
    # Plot Layout
    if log == True:
        layout = go.Layout(title = "Gráfico de dispersión (Escala Log.) <br> {0} - {1}".format(code.upper(), name),
                       xaxis = dict(title = 'Nivel simulado (cm)', type = 'log', ), 
                       yaxis = dict(title = 'Nivel observado (cm)', type = 'log', autorange = True), 
                       showlegend=True)
    else:
        layout = go.Layout(title = "Gráfico de dispersión <br> {0} - {1}".format(code.upper(), name),
                       xaxis = dict(title = 'Nivel simulado (cm)',  ), 
                       yaxis = dict(title = 'Nivel observado (cm)', autorange = True), 
                       showlegend=True)
    # Plotting data
    chart_obj = go.Figure(data=[scatter_data2, line_45], layout=layout)
    return(chart_obj)


# Metrics table
def get_metrics_table(merged_sim, merged_cor, my_metrics):
    # Metrics for corrected simulation data
    table_cor = hs.make_table(merged_cor, my_metrics)
    table_cor = table_cor.rename(index={'Full Time Series': 'Corrected Serie'})
    table_final = table_cor.transpose()
    # Merging data
    table_final = table_final.round(decimals=2)
    table_final = table_final.to_html(classes="table table-hover table-striped", table_id="corrected_1")
    table_final = table_final.replace('border="1"', 'border="0"').replace('<tr style="text-align: right;">','<tr style="text-align: left;">')
    return(table_final)


def _build_title(base, title_headers):
    if not title_headers:
        return base
    if 'bias_corrected' in title_headers.keys():
        base = 'Corrección del sesgo - ' + base
    for head in title_headers:
        if head == 'bias_corrected':
            continue
        base += f'<br>{head}: {title_headers[head]}'
    return base

def _plot_colors():
    return {
        '2 Year': 'rgba(254, 240, 1, .4)',
        '5 Year': 'rgba(253, 154, 1, .4)',
        '10 Year': 'rgba(255, 56, 5, .4)',
        '20 Year': 'rgba(128, 0, 246, .4)',
        '25 Year': 'rgba(255, 0, 0, .4)',
        '50 Year': 'rgba(128, 0, 106, .4)',
        '100 Year': 'rgba(128, 0, 246, .4)',
    }


def _rperiod_scatters(startdate: str, enddate: str, rperiods: pd.DataFrame, y_max: float, max_visible: float = 0,
                    visible: bool = None):
    colors = _plot_colors()
    x_vals = (startdate, enddate, enddate, startdate)
    r2 = rperiods['return_period_2'].values[0]
    if visible is None:
        if max_visible > r2:
            visible = True
        else:
            visible = 'legendonly'

    def template(name, y, color, fill='toself'):
        return go.Scatter(
            name=name,
            x=x_vals,
            y=y,
            legendgroup='returnperiods',
            fill=fill,
            visible=visible,
            line=dict(color=color, width=0))

    if list(rperiods.columns) == ['max_flow', 'return_period_20', 'return_period_10', 'return_period_2']:
        r10 = int(rperiods['return_period_10'].values[0])
        r20 = int(rperiods['return_period_20'].values[0])
        rmax = int(max(2 * r20 - r10, y_max))
        return [
            template(f'2 años: {r2}', (r2, r2, r10, r10), colors['2 Year']),
            template(f'10 años: {r10}', (r10, r10, r20, r20), colors['10 Year']),
            template(f'20 años: {r20}', (r20, r20, rmax, rmax), colors['20 Year']),
        ]

    else:
        r5 = int(rperiods['return_period_5'].values[0])
        r10 = int(rperiods['return_period_10'].values[0])
        r25 = int(rperiods['return_period_25'].values[0])
        r50 = int(rperiods['return_period_50'].values[0])
        r100 = int(rperiods['return_period_100'].values[0])
        rmax = int(max(2 * r100 - r25, y_max))
        return [
            template('Return Periods', (rmax, rmax, rmax, rmax), 'rgba(0,0,0,0)', fill='none'),
            template(f'2 años: {r2}', (r2, r2, r5, r5), colors['2 Year']),
            template(f'5 años: {r5}', (r5, r5, r10, r10), colors['5 Year']),
            template(f'10 años: {r10}', (r10, r10, r25, r25), colors['10 Year']),
            template(f'25 años: {r25}', (r25, r25, r50, r50), colors['25 Year']),
            template(f'50 años: {r50}', (r50, r50, r100, r100), colors['50 Year']),
            template(f'100 años: {r100}', (r100, r100, rmax, rmax), colors['100 Year']),
        ]



def forecast_stats(stats: pd.DataFrame, rperiods: pd.DataFrame = None, titles: dict = False,
                   outformat: str = 'plotly', hide_maxmin: bool = False) -> go.Figure:
    """
    Makes the streamflow data and optional metadata into a plotly plot

    Args:
        stats: the csv response from forecast_stats
        rperiods: the csv response from return_periods
        titles: (dict) Extra info to show on the title of the plot. For example:
            {'Reach ID': 1234567, 'Drainage Area': '1000km^2'}
        outformat: 'json', 'plotly', 'plotly_scatters', or 'plotly_html' (default plotly)
        hide_maxmin: Choose to hide the max/min envelope by default

    Return:
         plotly.GraphObject: plotly object, especially for use with python notebooks and the .show() method
    """

    # Start processing the inputs
    dates = stats.index.tolist()
    startdate = dates[0]
    enddate = dates[-1]

    plot_data = {
        'x_stats': stats['flow_avg_m^3/s'].dropna(axis=0).index.tolist(),
        'x_hires': stats['high_res_m^3/s'].dropna(axis=0).index.tolist(),
        'y_max': max(stats['flow_max_m^3/s']),
        'flow_max': list(stats['flow_max_m^3/s'].dropna(axis=0)),
        'flow_75%': list(stats['flow_75%_m^3/s'].dropna(axis=0)),
        'flow_avg': list(stats['flow_avg_m^3/s'].dropna(axis=0)),
        'flow_25%': list(stats['flow_25%_m^3/s'].dropna(axis=0)),
        'flow_min': list(stats['flow_min_m^3/s'].dropna(axis=0)),
        'high_res': list(stats['high_res_m^3/s'].dropna(axis=0)),
        'flow_med' : list(stats['flow_median_m^3/s'].dropna(axis=0)),
    }
    if rperiods is not None:
        plot_data.update(rperiods.to_dict(orient='index').items())
        max_visible = max(max(plot_data['flow_75%']), max(plot_data['flow_avg']), max(plot_data['high_res']))
        rperiod_scatters = _rperiod_scatters(startdate, enddate, rperiods, plot_data['y_max'], max_visible)
    else:
        rperiod_scatters = []

    maxmin_visible = 'legendonly' if hide_maxmin else True
    scatter_plots = [
        # Plot together so you can use fill='toself' for the shaded box, also separately so the labels appear
        go.Scatter(name='Nivel máximo y mínimo',
                   x=plot_data['x_stats'] + plot_data['x_stats'][::-1],
                   y=plot_data['flow_max'] + plot_data['flow_min'][::-1],
                   legendgroup='boundaries',
                   fill='toself',
                   visible=maxmin_visible,
                   line=dict(color='lightblue', dash='dash')),
        go.Scatter(name='Máximo',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_max'],
                   legendgroup='boundaries',
                   visible=maxmin_visible,
                   showlegend=False,
                   line=dict(color='darkblue', dash='dash')),
        go.Scatter(name='Mínimo',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_min'],
                   legendgroup='boundaries',
                   visible=maxmin_visible,
                   showlegend=False,
                   line=dict(color='darkblue', dash='dash')),

        go.Scatter(name='Percventil 25 - 75 del nivel',
                   x=plot_data['x_stats'] + plot_data['x_stats'][::-1],
                   y=plot_data['flow_75%'] + plot_data['flow_25%'][::-1],
                   legendgroup='percentile_flow',
                   fill='toself',
                   line=dict(color='lightgreen'), ),
        go.Scatter(name='75%',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_75%'],
                   showlegend=False,
                   legendgroup='percentile_flow',
                   line=dict(color='green'), ),
        go.Scatter(name='25%',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_25%'],
                   showlegend=False,
                   legendgroup='percentile_flow',
                   line=dict(color='green'), ),

        go.Scatter(name='Pronóstico de alta resolución',
                   x=plot_data['x_hires'],
                   y=plot_data['high_res'],
                   line={'color': 'black'}, ),
        go.Scatter(name='Nivel promedio del ensamble',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_avg'],
                   line=dict(color='blue'), ),
        go.Scatter(name='Mediana de caudal del ensamble',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_med'],
                   line=dict(color='cyan'), ),
    ]

    scatter_plots += rperiod_scatters

    layout = go.Layout(
        title=_build_title('Nivel pronosticado', titles),
        yaxis={'title': 'Nivel (cm)', 'range': [0, 'auto']},
        xaxis={'title': 'Fecha (UTC +0:00)', 'range': [startdate, enddate], 'hoverformat': '%H:%M - %b %d %Y',
               'tickformat': '%b %d %Y'},
    )
    figure = go.Figure(scatter_plots, layout=layout)

    return figure


# Forecast plot
def get_forecast_plot(comid, site, stats, rperiods, lowlevel_vals, records, obs_data):

    print(lowlevel_vals)

    corrected_stats_df = stats
    corrected_rperiods_df = rperiods
    fixed_records = records
    ##
    # hydroviewer_figure = geoglows.plots.forecast_stats(stats=corrected_stats_df,)
    hydroviewer_figure = forecast_stats(stats=corrected_stats_df,)
    layout = go.Layout(
        title = _build_title('Nivel de agua pronosticado', {'Estación': site, 'COMID': comid, 'bias_corrected': True}),
        yaxis = {'title': 'Nivel (cm)', 'range': [0, 'auto']},
    )

    hydroviewer_figure.update_layout(layout)
    x_vals = (corrected_stats_df.index[0], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[0])
    max_visible = max(corrected_stats_df.max())
    min_visible = min(corrected_stats_df.min())

    ##
    corrected_records_plot = fixed_records.loc[fixed_records.index >= pd.to_datetime(corrected_stats_df.index[0] - dt.timedelta(days=8))]
    corrected_records_plot = corrected_records_plot.loc[corrected_records_plot.index <= pd.to_datetime(corrected_stats_df.index[0] + dt.timedelta(days=2))]
    
    ##
    if len(corrected_records_plot.index) > 0:
      hydroviewer_figure.add_trace(go.Scatter(
          name='Pronóstico a 1 día',
          x=corrected_records_plot.index,
          y=corrected_records_plot.iloc[:, 0].values,
          line=dict(color='#FFA15A',)
      ))
      x_vals = (corrected_records_plot.index[0], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_records_plot.index[0])
      max_visible = max(max(corrected_records_plot.max()), max_visible)
      min_visible = min(max(corrected_records_plot.min()), min_visible)
    
    ## Getting Return Periods
    r2 = round(corrected_rperiods_df.iloc[0]['return_period_2'], 2)

    ## Colors
    colors = {
        '2 Year': 'rgba(254, 240, 1, .4)',
        '5 Year': 'rgba(253, 154, 1, .4)',
        '10 Year': 'rgba(255, 56, 5, .4)',
        '20 Year': 'rgba(128, 0, 246, .4)',
        '25 Year': 'rgba(255, 0, 0, .4)',
        '50 Year': 'rgba(128, 0, 106, .4)',
        '100 Year': 'rgba(128, 0, 246, .4)',
    }

    ##
    if max_visible > r2:
      visible = True
      hydroviewer_figure.for_each_trace(lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (), )
    else:
      visible = 'legendonly'
      hydroviewer_figure.for_each_trace(lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (), )
    
    ##
    def template(name, y, color, fill='toself', legendgroup='returnperiods'):
      return go.Scatter(
          name=name,
          x=x_vals,
          y=y,
          legendgroup=legendgroup,
          fill=fill,
          visible=visible,
          line=dict(color=color, width=0))
    ##
    r5 = round(corrected_rperiods_df.iloc[0]['return_period_5'], 2)
    r10 = round(corrected_rperiods_df.iloc[0]['return_period_10'], 2)
    r25 = round(corrected_rperiods_df.iloc[0]['return_period_25'], 2)
    r50 = round(corrected_rperiods_df.iloc[0]['return_period_50'], 2)
    r100 = round(corrected_rperiods_df.iloc[0]['return_period_100'], 2)
    ##
    hydroviewer_figure.add_trace(template('Periodo de retorno', (r100 * 0.05, r100 * 0.05, r100 * 0.05, r100 * 0.05), 'rgba(0,0,0,0)', fill='none'))
    hydroviewer_figure.add_trace(template(f'2 años: {r2}', (r2, r2, r5, r5), colors['2 Year']))
    hydroviewer_figure.add_trace(template(f'5 años: {r5}', (r5, r5, r10, r10), colors['5 Year']))
    hydroviewer_figure.add_trace(template(f'10 años: {r10}', (r10, r10, r25, r25), colors['10 Year']))
    hydroviewer_figure.add_trace(template(f'25 años: {r25}', (r25, r25, r50, r50), colors['25 Year']))
    hydroviewer_figure.add_trace(template(f'50 años: {r50}', (r50, r50, r100, r100), colors['50 Year']))
    hydroviewer_figure.add_trace(template(f'100 años: {r100}', (r100, r100, max(r100 + r100 * 0.05, max_visible), max(r100 + r100 * 0.05, max_visible)), colors['100 Year']))
    
    ##
    ## Fix axis in obs data for plot
    obs_data['fix_data'] = [data_i[data_i.index > corrected_records_plot.index[0]] for data_i in obs_data['data']]

    # Add observed data
    for num, data in enumerate(obs_data['fix_data']):
        hydroviewer_figure.add_trace(go.Scatter(
                                        name = obs_data['name'][num],
                                        x    = data.index,
                                        y    = data.iloc[:, 0].values,
                                        line = dict(color=obs_data['color'][num]),
                                        hovertemplate = 'Valor : %{y:.2f}<br>' + 
                                                        'Fecha : %{x|%Y/%m/%d}<br>' + 
                                                        'Hora : %{x|%H:%M}'
                                    ))

    # Add low data time serie to graph
    if np.nanmin(corrected_stats_df) < np.nanmin(lowlevel_vals.values):
        
        visible = True

        # Calc threshold value
        value_low_level = np.nanmin(lowlevel_vals.values)

        # Calc graph value
        min_forecast_data = np.nanmin(corrected_stats_df) * 0.8
        if min_forecast_data < 0:
            min_forecast_data = 1.05 * min_forecast_data

        # Add area to graph        
        hydroviewer_figure.add_trace(template(name = f'Umbral mínimo: {value_low_level:.2f}',
                                              y = (min_forecast_data, min_forecast_data, value_low_level, value_low_level),
                                              color = 'black',
                                              legendgroup = "lowlevels"))



    hydroviewer_figure['layout']['xaxis'].update(autorange=True)
    return(hydroviewer_figure)


def plot_historical_waterlevel(observed_df, corrected_df, station_code, station_name):

    # Fix database for plot
    observed_df = observed_df.dropna()
    corrected_df = corrected_df.dropna()

    # Plot
    observed_WL = go.Scatter(x=observed_df.index, y=observed_df.iloc[:, 0].values, name='Observado', line=dict(color="#636EFA"))
    corrected_WL = go.Scatter(x=corrected_df.index, y=corrected_df.iloc[:, 0].values, name='Simulado corregido', line=dict(color="#00CC96"))
    layout = go.Layout(
            title='Nivel de agua observado y simulado corregido <br> {0} - {1}'.format(station_code, station_name),
            xaxis=dict(title='Fechas', ), yaxis=dict(title='Nivel (cm)', autorange=True),
            showlegend=True)
    return(go.Figure(data=[observed_WL, corrected_WL], layout=layout))



####################################################################################################
##                                   CONTROLLERS AND REST APIs                                    ##
####################################################################################################

# Initialize the web app
@controller(name='home',url='national-water-level-forecast-colombia')
def home(request):
    context = {}
    return render(request, 'national_water_level_forecast_colombia/home.html', context)


# Return streamflow stations in geojson format 
@controller(name='get_stations',url='national-water-level-forecast-colombia/get-stations')
def get_stations(request):
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    try:
        # Query to database
        # stations = pd.read_sql("select *, concat(code, ' - ', left(name, 23)) from stations_waterlevel", conn)
        stations = pd.read_sql("select *, concat(codigo, ' - ', left(nombre, 23)) from stations_waterlevel order by codigo", conn)
    finally:
        conn.close()

    # Station filter
    stations.drop(["categoria", "tecnologia", "estado", "fecha_inst",
                   "altitud_x", "municipio", "area_opera", "area_hidro",
                   "entidad", "fecha_susp", "subzona_hi", "watershed",
                   "subbasin", "x", "y", "departamento", "zona_hidrografica"],
                   axis=1,
                   inplace = True)
    stations.rename(columns={"codigo"			: "code",
                            "nombre"			: "name",
                            "latitud"			: "latitude",
                            "longitud"			: "longitude",
                            "departamen"		: "loc3",
                            "zona_hidro"		: "basin",
                            "corriente"			: "river",
                            "comid"				: "comid",
                            "alert"				: "alert",
                            "altitud_y"			: "elevation",
                            "area_operativa"	: "loc1",
                            "area_hidrografica"	: "loc2",
                    }, inplace = True)
    stations = stations.astype({col : 'str' for col in stations.columns})
    stations = stations.astype({"latitude"  : 'float',
                                "longitude" : 'float'})
    
    stations = to_geojson(
        df = stations,
        lat = "latitude",
        lon = "longitude",
        properties = ["basin", "code", "name", "latitude", "longitude", "elevation", "comid", "river", 
                      "loc1", "loc2", "loc3", "alert", "concat"]
    )
    return JsonResponse(stations)


# Return streamflow station (in geojson format) 
@controller(name='get_data',url='national-water-level-forecast-colombia/get-data')
def get_data(request):
    # Retrieving GET arguments
    station_code = request.GET['codigo']
    station_comid = request.GET['comid']
    station_name = request.GET['nombre']
    plot_width = float(request.GET['width']) - 12
    plot_width_2 = 0.5*plot_width

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()

    # Data series
    observed_data = get_format_data("select datetime, s_{0} from observed_waterlevel_data order by datetime;".format(station_code), conn)
    simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
    # TODO : remove whwere geoglows server works
    simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()

    corrected_data = get_bias_corrected_data(simulated_data, observed_data)

    # Raw forecast
    ensemble_forecast = get_format_data("select * from f_{0};".format(station_comid), conn)
    forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    # return_periods = get_return_periods(station_comid, simulated_data)

    # Corrected forecast
    corrected_ensemble_forecast = __bias_correction_forecast__(simulated_data, ensemble_forecast, observed_data)
    corrected_forecast_records = get_corrected_forecast_records(forecast_records, simulated_data, observed_data)
    
    corrected_return_periods = get_return_periods(station_comid, corrected_data)
    lowlevel_vals = get_warning_low_level(station_comid, corrected_data)

    # FEWS data
    obs_fews, sen_fews = get_fews_data(station_code)

    # Stats for raw and corrected forecast
    # ensemble_stats = get_ensemble_stats(ensemble_forecast)
    corrected_ensemble_stats = get_ensemble_stats(corrected_ensemble_forecast)

    # Merge data (For plots)
    global merged_sim
    merged_sim = hd.merge_data(sim_df = simulated_data, obs_df = observed_data)
    global merged_cor
    merged_cor = hd.merge_data(sim_df = corrected_data, obs_df = observed_data)

    # Close conection
    conn.close()

    # Historical data plot
    corrected_data_plot = plot_historical_waterlevel(
                                observed_df = observed_data, 
                                corrected_df = corrected_data, 
                                station_code = station_code, 
                                station_name = station_name)
    
    # Daily averages plot
    daily_average_plot = get_daily_average_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code,
                                name = station_name)   
    # Monthly averages plot
    monthly_average_plot = get_monthly_average_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code,
                                name = station_name) 
    # Scatter plot
    data_scatter_plot = get_scatter_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code,
                                name = station_name,
                                log = False) 
    # Scatter plot (Log scale)
    log_data_scatter_plot = get_scatter_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code,
                                name = station_name,
                                log = True) 
    
    # Metrics table
    metrics_table = get_metrics_table(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                my_metrics = ["ME", "RMSE", "NRMSE (Mean)", "NSE", "KGE (2009)", "KGE (2012)", "R (Pearson)", "R (Spearman)", "r2"]) 
    
    # Percent of Ensembles that Exceed Return Periods    
    corrected_forecast_table = geoglows.plots.probabilities_table(
                                stats = corrected_ensemble_stats,
                                ensem = corrected_ensemble_forecast, 
                                rperiods = corrected_return_periods)

    # Ensemble forecast plot    
    corrected_ensemble_forecast_plot = get_forecast_plot(
                                            comid = station_comid, 
                                            site = station_name, 
                                            stats = corrected_ensemble_stats, 
                                            rperiods = corrected_return_periods,
                                            lowlevel_vals = lowlevel_vals, 
                                            records = corrected_forecast_records,
                                            obs_data = {'data'  : [obs_fews, sen_fews],
                                                        'color' : ['blue', 'red'],
                                                        'name'  : ['Nivel observado', 'Nivel sensor']})
    
    #returning
    context = {
        "corrected_data_plot": PlotlyView(corrected_data_plot.update_layout(width = plot_width)),
        "daily_average_plot": PlotlyView(daily_average_plot.update_layout(width = plot_width)),
        "monthly_average_plot": PlotlyView(monthly_average_plot.update_layout(width = plot_width)),
        "data_scatter_plot": PlotlyView(data_scatter_plot.update_layout(width = plot_width_2)),
        "log_data_scatter_plot": PlotlyView(log_data_scatter_plot.update_layout(width = plot_width_2)),
        "corrected_ensemble_forecast_plot": PlotlyView(corrected_ensemble_forecast_plot.update_layout(width = plot_width)),
        "metrics_table": metrics_table,
        "corrected_forecast_table": corrected_forecast_table,
    }
    return render(request, 'national_water_level_forecast_colombia/panel.html', context)





@controller(name='get_metrics_custom',url='national-water-level-forecast-colombia/get-metrics-custom')
def get_metrics_custom(request):
    # Combine metrics
    my_metrics_1 = ["ME", "RMSE", "NRMSE (Mean)", "NSE", "KGE (2009)", "KGE (2012)", "R (Pearson)", "R (Spearman)", "r2"]
    my_metrics_2 = request.GET['metrics'].split(",")
    lista_combinada = my_metrics_1 + my_metrics_2
    elementos_unicos = []
    elementos_vistos = set()
    for elemento in lista_combinada:
        if elemento not in elementos_vistos:
            elementos_unicos.append(elemento)
            elementos_vistos.add(elemento)
    # Compute metrics
    metrics_table = get_metrics_table(
                        merged_cor = merged_cor,
                        merged_sim = merged_sim,
                        my_metrics = elementos_unicos)
    return HttpResponse(metrics_table)



@controller(name='get_raw_forecast_date',url='national-water-level-forecast-colombia/get-raw-forecast-date')
def get_raw_forecast_date(request):
    ## Variables
    station_code = request.GET['codigo']
    station_comid = request.GET['comid']
    station_name = request.GET['nombre']
    forecast_date = request.GET['fecha']
    plot_width = float(request.GET['width']) - 12

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()

    # Data series
    observed_data = get_format_data("select datetime, s_{0} from observed_waterlevel_data order by datetime;".format(station_code), conn)
    simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
    # TODO : remove whwere geoglows server works
    simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()

    corrected_data = get_bias_corrected_data(simulated_data, observed_data)
    
    # Raw forecast
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    forecast_records = get_forecast_record_date(station_comid, forecast_date)
    return_periods = get_return_periods(station_comid, simulated_data)
    lowlevel_vals = get_warning_low_level(station_comid, simulated_data)

    # Corrected forecast
    corrected_ensemble_forecast = __bias_correction_forecast__(simulated_data, ensemble_forecast, observed_data)
    corrected_forecast_records = get_corrected_forecast_records(forecast_records, simulated_data, observed_data)
    corrected_return_periods = get_return_periods(station_comid, corrected_data)
    corrected_lowlevel_vals = get_warning_low_level(station_comid, corrected_data)
    
    # Forecast stats
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    corrected_ensemble_stats = get_ensemble_stats(corrected_ensemble_forecast)

    # Close conection
    conn.close()

    # FEWS data
    obs_fews, sen_fews = get_fews_data(station_code)
    
    # Plotting raw forecast
    ensemble_forecast_plot = get_forecast_plot(
                                comid = station_comid, 
                                site = station_name, 
                                stats = ensemble_stats, 
                                rperiods = return_periods, 
                                lowlevel_vals = lowlevel_vals,
                                records = forecast_records,
                                obs_data = {'data'  : [obs_fews, sen_fews],
                                                        'color' : ['blue', 'red'],
                                                        'name'  : ['Nivel observado', 'Nivel sensor']}).update_layout(width = plot_width).to_html()
    
    # Forecast table
    forecast_table = geoglows.plots.probabilities_table(
                                stats = ensemble_stats,
                                ensem = ensemble_forecast, 
                                rperiods = return_periods)


    # Plotting corrected forecast
    corr_ensemble_forecast_plot = get_forecast_plot(
                                    comid = station_comid, 
                                    site = station_name, 
                                    stats = corrected_ensemble_stats, 
                                    rperiods = corrected_return_periods, 
                                    records = corrected_forecast_records,
                                    lowlevel_vals = corrected_lowlevel_vals,
                                    obs_data = {'data'  : [obs_fews, sen_fews],
                                                        'color' : ['blue', 'red'],
                                                        'name'  : ['Nivel observado', 'Nivel sensor']}).update_layout(width = plot_width).to_html()
    
    # Corrected forecast table
    corr_forecast_table = geoglows.plots.probabilities_table(
                                    stats = corrected_ensemble_stats,
                                    ensem = corrected_ensemble_forecast, 
                                    rperiods = corrected_return_periods)
    
    return JsonResponse({
       'ensemble_forecast_plot': ensemble_forecast_plot,
       'forecast_table': forecast_table,
       'corr_ensemble_forecast_plot': corr_ensemble_forecast_plot,
       'corr_forecast_table': corr_forecast_table
    })
    

    # return render(request, 'national_water_level_forecast_colombia/home.html', context)

############################################################
# Retrieve observed data
@controller(name='get_observed_data_xlsx',
            url='national-water-level-forecast-colombia/get-observed-data-xlsx')
def get_observed_data_xlsx(request):
    
    # Retrieving GET arguments
    station_code  = request.GET['codigo']
    station_comid = request.GET['comid']
    forecast_date = request.GET['fecha']
    
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    
    try:
    # Data series
        data = get_format_data("select datetime, s_{0} from observed_waterlevel_data order by datetime;".format(station_code), conn)
        data.rename(columns={'s_{0}'.format(station_code): "Historical observation (cm)"}, inplace=True)
    finally:
       conn.close()
    
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='serie_observada_simulada', index=True)
    writer.save()
    output.seek(0)

    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=serie_historica_observada.xlsx'
    response.write(output.getvalue())
    return response



# Retrieve simualted corrected data
@controller(name='get_corrected_data_xlsx',
            url='national-water-level-forecast-colombia/get-corrected-data-xlsx')
def get_corrected_data_xlsx(request):

    # Retrieving GET arguments
    station_code = request.GET['codigo']
    station_comid = request.GET['comid']
    forecast_date = request.GET['fecha']
    
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    
    try:
        # Data series
        observed_data  = get_format_data("select datetime, s_{0} from observed_waterlevel_data order by datetime;".format(station_code), conn)
        simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
        # TODO : remove whwere geoglows server works
        simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()
    finally:
       conn.close()

    # Fix data
    data = get_bias_corrected_data(simulated_data, observed_data)
    data.rename(columns={"Corrected Simulated Streamflow" : "Corrected Simulated Waterlevel (cm)"}, 
                inplace = True)
    
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='serie_historica_corregida', index=True)
    writer.save()
    output.seek(0)

    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=serie_historica_corregida.xlsx'
    response.write(output.getvalue())
    return response



@controller(name='get_corrected_forecast_xlsx',
            url='national-water-level-forecast-colombia/get-corrected-forecast-xlsx')
def get_corrected_forecast_xlsx(request):
    # Retrieving GET arguments
    station_code = request.GET['codigo']
    station_comid = request.GET['comid']
    forecast_date = request.GET['fecha']

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    try:
        # Data series
        observed_data  = get_format_data("select datetime, s_{0} from observed_waterlevel_data order by datetime;".format(station_code), conn)
        simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
        # TODO : remove whwere geoglows server works
        simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()
        forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    finally:
        conn.close()

    # Raw forecast
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    
    # Corrected forecast
    corrected_ensemble_forecast = __bias_correction_forecast__(simulated_data, ensemble_forecast, observed_data)
    
    # Forecast stats
    corrected_ensemble_stats = get_ensemble_stats(corrected_ensemble_forecast)
    
    # Forecast record correctes
    corrected_forecast_records = get_corrected_forecast_records(forecast_records, simulated_data, observed_data)    

    # Fix column names
    fix_label = lambda x : x.replace('flow', 'waterlevel').replace('m^3/s', 'cm')
    corrected_ensemble_stats.rename(columns={col : fix_label(col) for col in corrected_ensemble_stats.columns}, inplace=True)
    corrected_ensemble_forecast.rename(columns={col : fix_label(col) for col in corrected_ensemble_forecast.columns}, inplace=True)
    corrected_forecast_records.rename(columns={col : fix_label(col) for col in corrected_forecast_records.columns}, inplace=True)

    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    corrected_ensemble_stats.to_excel(writer, sheet_name='corrected_ensemble_stats', index=True)
    corrected_ensemble_forecast.to_excel(writer, sheet_name='corrected_ensemble_forecast', index=True)
    corrected_forecast_records.to_excel(writer, sheet_name='corrected_forecast_records', index=True)
    writer.save()
    output.seek(0)

    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=corrected_ensemble_forecast.xlsx'
    response.write(output.getvalue())
    return response



############################################################
@controller(name = "user_manual",
            url  = "national-water-level-forecast-colombia/user_manual")
def user_manual(request):
    context = {}
    return render(request, 'national_water_level_forecast_colombia/user_manual.html', context)


@controller(name = "technical_manual",
            url  = "national-water-level-forecast-colombia/technical_manual")
def technical_manual(request):
    context = {}
    return render(request, 'national_water_level_forecast_colombia/technical_manual.html', context)


############################################################
#                          SERVICES                        #
############################################################
@api_view(['GET'])
@authentication_classes((TokenAuthentication,))
@controller(name='get_image',
            url='national-water-level-forecast-colombia/get-image',
            login_required=False)
def download_img(request):

    # Retrieving GET arguments
    station_code  = request.GET['codigo']
    type_graph = request.GET['typeGraph']

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()

    try:
        # Stations dataframe
        stations = pd.read_sql("select codigo, nombre, comid from stations_waterlevel order by codigo", conn)
        # Read input data
        input_df = stations[stations['codigo'] == station_code].dropna().reset_index(drop=True)
        # Extrac data
        station_comid = str(int(input_df['comid'].values[0]))
        station_name  = input_df['nombre'].values[0]

        # Data series
        observed_data = get_format_data("select datetime, s_{0} from observed_waterlevel_data order by datetime;".format(station_code), conn)
        simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
        # TODO : remove whwere geoglows server works
        simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()
        corrected_data = get_bias_corrected_data(simulated_data, observed_data)

        # Raw forecast
        ensemble_forecast = get_format_data("select * from f_{0};".format(station_comid), conn)
        forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
        
    finally:
        # Close conection
        conn.close()

    # Returns periods
    # return_periods = get_return_periods(station_comid, simulated_data)

    # Corrected forecast
    corrected_ensemble_forecast = __bias_correction_forecast__(simulated_data, ensemble_forecast, observed_data)
    corrected_forecast_records = get_corrected_forecast_records(forecast_records, simulated_data, observed_data)
    corrected_return_periods = get_return_periods(station_comid, corrected_data)
    corrected_lowlevel_vals = get_warning_low_level(station_comid, corrected_data)

    # FEWS data
    obs_fews, sen_fews = get_fews_data(station_code)

    # Stats for raw and corrected forecast
    # ensemble_stats = get_ensemble_stats(ensemble_forecast)
    corrected_ensemble_stats = get_ensemble_stats(corrected_ensemble_forecast)

    # Merge data (For plots)
    global merged_sim
    merged_sim = hd.merge_data(sim_df = simulated_data, obs_df = observed_data)
    global merged_cor
    merged_cor = hd.merge_data(sim_df = corrected_data, obs_df = observed_data)

    # PLOT
    # Ensemble forecast plot
    if 'historical' == type_graph:
        fig = plot_historical_waterlevel(
                    observed_df = observed_data, 
                    corrected_df = corrected_data, 
                    station_code = station_code, 
                    station_name = station_name)
        name_file = 'historical'
    elif  'forecast' == type_graph:
        fig = get_forecast_plot(comid = station_comid, 
                                site = station_name, 
                                stats = corrected_ensemble_stats, 
                                rperiods = corrected_return_periods, 
                                lowlevel_vals = corrected_lowlevel_vals,
                                records = corrected_forecast_records,
                                obs_data = {'data'  : [obs_fews, sen_fews],
                                            'color' : ['blue', 'red'],
                                            'name'  : ['Nivel observado', 'Nivel sensor']})
        name_file = 'corrected_forecast'

    # Build image bytes
    img_bytes = fig.to_image(format="png")

    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type="image/jpeg")
    response['Content-Disposition'] = 'attachment; filename={0}_{1}_H.png'.format(name_file, station_code)
    response.write(img_bytes)

    return response


@api_view(['GET'])
@authentication_classes((TokenAuthentication,))
@controller(name='get_warning',
            url='national-water-level-forecast-colombia/get-warnings',
            login_required=False)
def down_load_warning_list(request):

    # stations_streamflow
    # stations_waterlevel

    db= create_engine(tokencon)
    conn = db.connect()
    try:
        df = pd.read_sql("select codigo, nombre, comid, alert from stations_waterlevel order by codigo", conn)
    finally:
        # Close conection
        conn.close()
    
    return JsonResponse(df.to_dict(orient='list'))