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
import requests
import geoglows
import numpy as np
import HydroErr as he
import datetime as dt
import hydrostats as hs
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

def gumbel_1(std: float, xbar: float, rp: int or float) -> float:
    return -math.log(-math.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)


def get_return_periods(comid, data):
    # Stats
    max_annual_flow = data.groupby(data.index.strftime("%Y")).max()
    mean_value = np.mean(max_annual_flow.iloc[:,0].values)
    std_value = np.std(max_annual_flow.iloc[:,0].values)
    # Return periods
    return_periods = [100.0, 50.0, 25.0, 10.0, 5.0, 2.0]
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


def ensemble_quantile(ensemble, quantile, label):
    df = ensemble.quantile(quantile, axis=1).to_frame()
    df.rename(columns = {quantile: label}, inplace = True)
    return(df)

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
        high_res_df
    ], axis=1)
    return(stats_df)



def get_forecast_date(comid, date):
    url = 'https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id={0}&date={1}&return_format=csv'.format(comid, date)
    status = False
    cnt = 0
    while not status:
      try:
        outdf = pd.read_csv(url, index_col=0)
        status = True
      except:
        print("Trying to retrieve data...")
        if cnt > 10:
            print('Error in request forecast for download')
            outdf = pd.DataFrame(data={'ensemble_52_m^3/s' : [0]})
            status = True
        cnt += 1
    # Filter and correct data
    try:
        outdf[outdf < 0] = 0
        outdf.index = pd.to_datetime(outdf.index)
        outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        outdf.index = pd.to_datetime(outdf.index)
    finally:
        return outdf


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


def get_fews_data(station_code, folder, file_subfix):
    # TODO : Change with fews postgres implementation
    url = 'http://fews.ideam.gov.co/colombia/{}/00'.format(folder)\
          + station_code + '{}.json'.format(file_subfix)
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
                                                  'data' : observedDischarge})
        observedDischarge_df['date'] = pd.to_datetime(observedDischarge_df['date'], format='%Y/%m/%d %H:%M')
        observedDischarge_df['data'] = observedDischarge_df['data'].astype(float)
        observedDischarge_df.dropna(inplace=True)
        observedDischarge_df.set_index('date', inplace = True)
        
        # Build dataframe sensor
        sensorDischarge_df   = pd.DataFrame(data={'date' : datesSensorDischarge,
                                                'data' : sensorDischarge})
        sensorDischarge_df['date'] = pd.to_datetime(sensorDischarge_df['date'], format='%Y/%m/%d %H:%M')
        sensorDischarge_df['data'] = sensorDischarge_df['data'].astype(float)
        sensorDischarge_df.dropna(inplace=True)
        sensorDischarge_df.set_index('date', inplace=True)

    except:
        # Build discharge dataframe
        observedDischarge_df = pd.DataFrame(data = {'date' : [pd.NaT],
                                                  'data' : [np.nan]})
        observedDischarge_df.set_index('date', inplace = True)

        # Build sensor dataframe
        sensorDischarge_df = pd.DataFrame(data = {'date' : [pd.NaT],
                                                  'data' : [np.nan]})
        sensorDischarge_df.set_index('date', inplace=True)

    return observedDischarge_df, sensorDischarge_df

####################################################################################################
##                                      PLOTTING FUNCTIONS                                        ##
####################################################################################################

# Plotting daily averages values
def get_daily_average_plot(sim, comid):
    # Generate the average values
    daily_avg_sim = hd.daily_average(sim)
    # Generate the plots on Ploty
    daily_avg_sim_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 0].values, name = 'Simulado', )
    # PLot Layout
    layout = go.Layout(
        title='Caudal promedio diario <br>COMID: {0}'.format(comid),
        xaxis=dict(title='Días', ), 
        yaxis=dict(title='Caudal (m<sup>3</sup>/s)', autorange=True),
        showlegend=False)
    # Generate the output
    chart_obj = go.Figure(data=[daily_avg_sim_Q], layout=layout)
    return(chart_obj)


# Plotting monthly averages values
def get_monthly_average_plot(sim, comid):
    # Generate the average values
    daily_avg_sim = hd.monthly_average(sim)
    # Generate the plots on Ploty
    daily_avg_sim_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 0].values, name = 'Simulado', )
    # PLot Layout
    layout = go.Layout(
        title='Caudal medio mensual <br>COMID: {0}'.format(comid),
        xaxis=dict(title='Mes', ), 
        yaxis=dict(title='Caudal (m<sup>3</sup>/s)', autorange=True),
        showlegend=False)
    # Generate the output
    chart_obj = go.Figure(data=[daily_avg_sim_Q], layout=layout)
    return(chart_obj)


# Acumulate volume
def get_acumulated_volume_plot(sim, comid):
    # Parse dataframe to array
    sim_array = sim.iloc[:, 0].values * 0.0864
    # Convert from m3/s to Hm3
    sim_volume = sim_array.cumsum()
    # Generate plots
    simulated_volume = go.Scatter(x = sim.index, y = sim_volume, name='Simulado', )
    # Plot layouts
    layout = go.Layout(
                title='Curva de columen acumulada simulada <br>{0}'.format(comid),
                xaxis=dict(title='Fecha', ), 
                yaxis=dict(title='Volumen (Mm<sup>3</sup>)', autorange=True),
                showlegend=False)
    # Integrating the plots
    chart_obj = go.Figure(data=[simulated_volume], layout=layout)
    return(chart_obj)



# Forecast plot
def get_forecast_plot(comid, stats, rperiods, records):
    corrected_stats_df = stats
    corrected_rperiods_df = rperiods
    fixed_records = records
    ##
    # hydroviewer_figure = geoglows.plots.forecast_stats(stats=corrected_stats_df, titles={'COMID': comid})
    hydroviewer_figure = forecast_stats(stats=corrected_stats_df, titles={'COMID': comid})
    x_vals = (corrected_stats_df.index[0], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[0])
    max_visible = max(corrected_stats_df.max())
    ##
    corrected_records_plot = fixed_records.loc[fixed_records.index >= pd.to_datetime(corrected_stats_df.index[0] - dt.timedelta(days=8))]
    corrected_records_plot = corrected_records_plot.loc[corrected_records_plot.index <= pd.to_datetime(corrected_stats_df.index[0] + dt.timedelta(days=2))]
    ##
    if len(corrected_records_plot.index) > 0:
      hydroviewer_figure.add_trace(go.Scatter(
          name='Pronostico a 1 día',
          x=corrected_records_plot.index,
          y=corrected_records_plot.iloc[:, 0].values,
          line=dict(color='#FFA15A',)
      ))
      x_vals = (corrected_records_plot.index[0], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_records_plot.index[0])
      max_visible = max(max(corrected_records_plot.max()), max_visible)

    ## Getting Return Periods
    r2 = round(corrected_rperiods_df.iloc[0]['return_period_2'], 1)
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
    def template(name, y, color, fill='toself'):
      return go.Scatter(
          name=name,
          x=x_vals,
          y=y,
          legendgroup='returnperiods',
          fill=fill,
          visible=visible,
          line=dict(color=color, width=0))
    
    ##
    r5 = round(corrected_rperiods_df.iloc[0]['return_period_5'], 1)
    r10 = round(corrected_rperiods_df.iloc[0]['return_period_10'], 1)
    r25 = round(corrected_rperiods_df.iloc[0]['return_period_25'], 1)
    r50 = round(corrected_rperiods_df.iloc[0]['return_period_50'], 1)
    r100 = round(corrected_rperiods_df.iloc[0]['return_period_100'], 1)
    ##
    hydroviewer_figure.add_trace(template('Periodo de retorno', (r100 * 0.05, r100 * 0.05, r100 * 0.05, r100 * 0.05), 'rgba(0,0,0,0)', fill='none'))
    hydroviewer_figure.add_trace(template(f'2 años: {r2:.2f}', (r2, r2, r5, r5), colors['2 Year']))
    hydroviewer_figure.add_trace(template(f'5 años: {r5:.2f}', (r5, r5, r10, r10), colors['5 Year']))
    hydroviewer_figure.add_trace(template(f'10 años: {r10:.2f}', (r10, r10, r25, r25), colors['10 Year']))
    hydroviewer_figure.add_trace(template(f'25 años: {r25:.2f}', (r25, r25, r50, r50), colors['25 Year']))
    hydroviewer_figure.add_trace(template(f'50 años: {r50:.2f}', (r50, r50, r100, r100), colors['50 Year']))
    hydroviewer_figure.add_trace(template(f'100 años: {r100:.2f}', (r100, r100, max(r100 + r100 * 0.05, max_visible), max(r100 + r100 * 0.05, max_visible)), colors['100 Year']))
    ##
    hydroviewer_figure['layout']['xaxis'].update(autorange=True)
    return(hydroviewer_figure)


# FEWS plot
def plot_fews_time_series(data, title, axis_names, hlines : dict = None):

    # Build image
    fews_plot = go.Figure()

    # Add tracers
    x_list = []
    for data_i, color_i, name_i in list(zip(data['data'], data['color'], data['name'])):
        fews_plot.add_trace(go.Scatter(x = data_i.index,
                                       y = data_i[data_i.columns].values.flatten(),
                                       mode = 'lines+markers',
                                       marker_color=color_i,
                                       name = name_i,
                                       connectgaps=False,
                                       legendgroup='G1',
                                       legendgrouptitle_text="Método de obtención",
                                       hovertemplate = 'Valor : %{y:.2f}<br>' + 
                                                       'Fecha : %{x|%Y/%m/%d}<br>' + 
                                                       'Hora : %{x|%H:%M}'))
        x_list += data_i.index.tolist()

    if 0 >= len(x_list):
       x_list = [np.nan]

    x_ini = np.min(x_list)
    x_end = np.max(x_list)
    
    if hlines is not None:   
        # Add hlines
        for data_i, color_i, name_i in list(zip(hlines['data'], hlines['color'], hlines['name'])):

            if not np.isnan(data_i):
                name_line = '{} - {:.2f} {}'.format(name_i, data_i, hlines['units'])
            else:
                name_line = name_i
            
            fews_plot.add_trace(go.Scatter(x = [x_ini, x_end],
                                           y = [data_i, data_i],
                                           mode = 'lines',
                                           marker_color=color_i,
                                           name = name_line,
                                           connectgaps=False,
                                           legendgroup='G2',
                                           legendgrouptitle_text="Umbrales"))
    
    fews_plot.update_layout(title=title,
                            yaxis_title=axis_names[0],
                            xaxis_title=axis_names[1],
                            legend_title="Leyenda")
   
    return fews_plot

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
        r10 = rperiods['return_period_10'].values[0]
        r20 = rperiods['return_period_20'].values[0]
        rmax = max(2 * r20 - r10, y_max)
        return [
            template(f'2 años: {round(r2,1):.2f}', (r2, r2, r10, r10), colors['2 Year']),
            template(f'10 años: {round(r10,1):.2f}', (r10, r10, r20, r20), colors['10 Year']),
            template(f'20 años: {round(r20,1):.2f}', (r20, r20, rmax, rmax), colors['20 Year']),
        ]

    else:
        r5 = rperiods['return_period_5'].values[0]
        r10 = rperiods['return_period_10'].values[0]
        r25 = rperiods['return_period_25'].values[0]
        r50 = rperiods['return_period_50'].values[0]
        r100 = rperiods['return_period_100'].values[0]
        rmax = max(2 * r100 - r25, y_max)

        return [
            template('Periodos de retorno', (rmax, rmax, rmax, rmax), 'rgba(0,0,0,0)', fill='none'),
            template(f'2 años: {round(r2,1):.2f}', (r2, r2, r5, r5), colors['2 Year']),
            template(f'5 años: {round(r5,1):.2f}', (r5, r5, r10, r10), colors['5 Year']),
            template(f'10 años: {round(r10,1):.2f}', (r10, r10, r25, r25), colors['10 Year']),
            template(f'25 años: {round(r25,1):.2f}', (r25, r25, r50, r50), colors['25 Year']),
            template(f'50 años: {round(r50,1):.2f}', (r50, r50, r100, r100), colors['50 Year']),
            template(f'100 años: {round(r100,1):.2f}', (r100, r100, rmax, rmax), colors['100 Year']),
        ]


# PLOTTING AUXILIARY FUNCTIONS
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


def historic_simulation(hist: pd.DataFrame, rperiods: pd.DataFrame = None, titles: dict = False,
                        outformat: str = 'plotly') -> go.Figure:
    """

    Adapted from geoglows library

    Makes the streamflow ensemble data and metadata into a plotly plot

    Args:
        hist: the csv response from historic_simulation
        rperiods: the csv response from return_periods
        outformat: either 'json', 'plotly', or 'plotly_html' (default plotly)
        titles: (dict) Extra info to show on the title of the plot. For example:
            {'Reach ID': 1234567, 'Drainage Area': '1000km^2'}

    Return:
         plotly.GraphObject: plotly object, especially for use with python notebooks and the .show() method
    """

    dates = hist.index.tolist()
    startdate = dates[0]
    enddate = dates[-1]

    plot_data = {
        'x_datetime': dates,
        'y_flow': hist.values.flatten(),
        'y_max': max(hist.values.flatten()),
    }
    if rperiods is not None:
        plot_data.update(rperiods.to_dict(orient='index').items())
        rperiod_scatters = _rperiod_scatters(startdate, enddate, rperiods, plot_data['y_max'], plot_data['y_max'])
    else:
        rperiod_scatters = []

    scatter_plots = [go.Scatter(
        name='Simulación histórica',
        x=plot_data['x_datetime'],
        y=plot_data['y_flow'])
    ]
    scatter_plots += rperiod_scatters

    layout = go.Layout(
        title=_build_title('Caudal simulado histórico', titles),
        yaxis={'title': 'Caudal (m<sup>3</sup>/s)', 'range': [0, 'auto']},
        xaxis={'title': 'Fecha (UTC +0:00)', 'range': [startdate, enddate], 'hoverformat': '%b %d %Y',
               'tickformat': '%Y'},
    )
    figure = go.Figure(scatter_plots, layout=layout)
    
    return figure


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
        go.Scatter(name='Caudal máximo y mínimo',
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

        go.Scatter(name='Percventil 25 - 75 de caudal',
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
        go.Scatter(name='Caudal promedio del ensamble',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_avg'],
                   line=dict(color='blue'), ),
    ]

    scatter_plots += rperiod_scatters

    layout = go.Layout(
        title=_build_title('Caudal pronosticado', titles),
        yaxis={'title': 'Caudal (m<sup>3</sup>/s)', 'range': [0, 'auto']},
        xaxis={'title': 'Fecha (UTC +0:00)', 'range': [startdate, enddate], 'hoverformat': '%b %d %Y',
               'tickformat': '%b %d %Y'},
    )
    figure = go.Figure(scatter_plots, layout=layout)

    return figure



####################################################################################################
##                                   CONTROLLERS AND REST APIs                                    ##
####################################################################################################

# Initialize the web app
@controller(name='home', url='hydroviewer-colombia')
def home(request):
    return render(request, 'hydroviewer_colombia/home.html')


# Return data and plots for the selected comid
@controller(name='get_data_fews', url='hydroviewer-colombia/get-data-fews')
def get_data_fews(request):
    # Retrieving GET arguments
    code      = request.GET['code']
    umaxhis   = float(request.GET['umaxhis'])
    ubajos    = float(request.GET['ubajos'])
    uamarilla = float(request.GET['uamarilla'])
    unaranja  = float(request.GET['unaranja'])
    uroja     = float(request.GET['uroja'])

    plot_width = float(request.GET['width']) - 12

    # Load data
    obs_st, sen_st = get_fews_data(code, 'jsonQ', 'Qobs')
    obs_wl, sen_wl = get_fews_data(code, 'jsonH', 'Hobs')

    # Plot results
    streamflow_plot  = plot_fews_time_series(data  = {'data'  : [obs_st, sen_st],
                                                      'color' : ['green', 'blue'],
                                                      'name'  : ['Observado' , 'Sensor']},
                                             title = 'Caudal observado en tiempo real. <br>[Estación : {}]'.format(code),
                                             axis_names = ['Caudal m3/s', 'fecha'],
                                             )
    
    waterlevel_plot = plot_fews_time_series(data  = {'data'  : [obs_wl, sen_wl],
                                                     'color' : ['green', 'blue'],
                                                     'name'  : ['Observado' , 'Sensor']},
                                            hlines = {'data'  : [umaxhis, ubajos, uamarilla, unaranja, uroja],
                                                      'color' : ['black', 'purple', 'yellow', 'orange', 'red'],
                                                      'name'  : ['Máximos', 'Bajos', 'Amarilla', 'Naranja', 'Roja'],
                                                      'units' : 'm'},
                                            title = 'Nivel observado en tiempo real. <br>[Estación : {}]'.format(code),
                                            axis_names = ['Nivel m', 'fecha']
                                            )

    context = {
       "streamflow_plot": PlotlyView(streamflow_plot.update_layout(width = plot_width)),
       "waterlevel_plot": PlotlyView(waterlevel_plot.update_layout(width = plot_width))
    }

    return render(request, 'hydroviewer_colombia/panel_fews.html', context) 


# Return data and plots for the selected comid
@controller(name='get_data', url='hydroviewer-colombia/get-data')
def get_data(request):

    # Retrieving GET arguments
    station_comid = request.GET['comid']

    plot_width = float(request.GET['width']) - 12
    plot_width_2 = 0.5 * plot_width

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()

    # Data series
    simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
    # TODO : remove whwere geoglows server works
    simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()

    ensemble_forecast = get_format_data("select * from f_{0};".format(station_comid), conn)
    forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    return_periods = get_return_periods(station_comid, simulated_data)

    # Close conection
    conn.close()

    # Plots and tables
    """
    data_plot = geoglows.plots.historic_simulation(
                                    hist = simulated_data,
                                    rperiods = return_periods,
                                    outformat = "plotly",
                                    titles = {'Reach COMID': station_comid})
    """
    data_plot = historic_simulation(
                                    hist = simulated_data,
                                    rperiods = return_periods,
                                    outformat = "plotly",
                                    titles = {'COMID': station_comid})
    daily_average_plot = get_daily_average_plot(
                                    sim = simulated_data,
                                    comid = station_comid)
    monthly_average_plot = get_monthly_average_plot(
                                    sim = simulated_data,
                                    comid = station_comid)
    flow_duration_curve = geoglows.plots.flow_duration_curve(
                                    hist = simulated_data,
                                    outformat = "plotly",
                                    titles = {'COMID': station_comid})
    acumulated_volume_plot = get_acumulated_volume_plot(
                                    sim = simulated_data,
                                    comid = station_comid)
    forecast_plot = get_forecast_plot(
                                    comid = station_comid, 
                                    stats = ensemble_stats, 
                                    rperiods = return_periods, 
                                    records = forecast_records)
    forecast_table = geoglows.plots.probabilities_table(
                                    stats = ensemble_stats,
                                    ensem = ensemble_forecast, 
                                    rperiods = return_periods)

    #returning
    context = {
        "data_plot": PlotlyView(data_plot.update_layout(width = plot_width)),
        "daily_average_plot": PlotlyView(daily_average_plot.update_layout(width = plot_width)),
        "monthly_average_plot": PlotlyView(monthly_average_plot.update_layout(width = plot_width)),
        "flow_duration_curve": PlotlyView(flow_duration_curve.update_layout(width = plot_width_2)),
        "acumulated_volume_plot": PlotlyView(acumulated_volume_plot.update_layout(width = plot_width_2)),
        "forecast_plot": PlotlyView(forecast_plot.update_layout(width = plot_width)),
        "forecast_table": forecast_table,
    }
    return render(request, 'hydroviewer_colombia/panel.html', context) 


# Return streamflow station (in geojson format) 
@controller(name='get_raw_forecast_date',url='hydroviewer-colombia/get-raw-forecast-date')
def get_raw_forecast_date(request):

    # Retrieving GET arguments
    station_comid = request.GET['comid']
    plot_width    = float(request.GET['width']) - 12
    forecast_date = request.GET['fecha']

    # Establish connection to database
    db = create_engine(tokencon)
    conn = db.connect()

    # Data series
    simulated_data    = get_format_data("select * from hs_{0};".format(station_comid), conn)
    # TODO : remove whwere geoglows server works
    simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()

    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    forecast_records  = get_forecast_record_date(station_comid, forecast_date)
    ensemble_stats    = get_ensemble_stats(ensemble_forecast)
    return_periods    = get_return_periods(station_comid, simulated_data)

    # Close conection
    conn.close()

    # Plots and tables
    forecast_plot = get_forecast_plot(comid = station_comid, 
                                      stats = ensemble_stats, 
                                      rperiods = return_periods, 
                                      records = forecast_records)
    
    forecast_table = geoglows.plots.probabilities_table(
                                    stats = ensemble_stats,
                                    ensem = ensemble_forecast, 
                                    rperiods = return_periods)

    # Returning
    context = {
        "forecast_plot": PlotlyView(forecast_plot.update_layout(width = plot_width)),
        "forecast_table": forecast_table,
    }
    return render(request, 'hydroviewer_colombia/forecast_panel.html', context)


@controller(name = 'get_fews_alerts', 
            url  = 'hydroviewer-colombia/get-fews-alerts')
def get_fews_alerts(request):
   # Establish connection to database
    db = create_engine(tokencon)
    conn = db.connect()
    try:
        # Query to database
        stations = pd.read_sql("select * from stations_fews", conn)
    finally:
       conn.close()
    db.dispose()
    
    # DB mannagement
    for col in stations.columns:
        stations[col] = stations[col].astype('str')
    stations['lat'] = stations['lat'].astype('float')
    stations['lng'] = stations['lng'].astype('float')

    stations = to_geojson(
        df = stations,
        lat = "lat",
        lon = "lng",
        properties = ['id', 'nombre', 'lng', 'lat', 'altitud', 'corriente', 'subzona', 'zona',
                      'cenpoblado', 'municipio', 'ctg', 'cotacero', 'areaaferete', 'depart',
                      'alert', 'uroja', 'unaranja', 'uamarilla', 'ubajos', 'umaxhis',
                      'umbralsen', 'umbralobs', 'ultimonivelsen', 'ultimonivelobs']
    )

    return JsonResponse(stations)


# Return alerts (in geojson format)
@controller(name='get_alerts',url='hydroviewer-colombia/get-alerts')
def get_alerts(request):
    # Establish connection to database
    db = create_engine(tokencon)
    conn = db.connect()

    # Query to database
    stations = pd.read_sql("select * from drainage where alert != 'R0'", conn)

    # Fix for station database
    stations.rename(columns={'hydroid'    : 'comid',
                             'dpto_cnmbr' : 'loc0',
                             'nom_zh'     : 'loc1',
                             'nom_szh'    : 'loc2',
                             'nombre_mpi' : 'loc3',
                             'nom_ah'     : 'loc4'}, inplace = True)

    for col in stations.columns:
        stations[col] = stations[col].astype('str')

    stations['latitude']  = stations['latitude'].astype('float')
    stations['longitude'] = stations['longitude'].astype('float')

    conn.close()

    stations = to_geojson(
        df = stations,
        lat = "latitude",
        lon = "longitude",
        properties = ["comid", "latitude", "longitude", "river", "loc0", "loc1", "loc2", "loc3", "loc4", "alert"]
    )
    return JsonResponse(stations)


# Return alerts (in geojson format)
@controller(name='get_rivers',url='hydroviewer-colombia/get-rivers')
def get_rivers(request):
    # Establish connection to database
    db = create_engine(tokencon)
    conn = db.connect()
    
    # Query to database
    stations = pd.read_sql("select * from drainage", conn)

    # Fix database
    stations.rename(columns={'hydroid'    : 'comid',
                             'dpto_cnmbr' : 'loc0',
                             'nom_zh'     : 'loc1',
                             'nom_szh'    : 'loc2'}, inplace = True)
    
    for col in stations.columns:
        stations[col] = stations[col].astype('str')
    
    stations['latitude']  = stations['latitude'].astype('float')
    stations['longitude'] = stations['longitude'].astype('float')

    conn.close()
    stations = to_geojson(
        df = stations,
        lat = "latitude",
        lon = "longitude",
        properties = ["comid", "latitude", "longitude", "river", "loc0", "loc1", "loc2", "alert"]
    )
    return JsonResponse(stations)




# Retrieve xlsx data
@controller(name='get_simulated_data_xlsx',url='hydroviewer-colombia/get-simulated-data-xlsx')
def get_simulated_data_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid'] #9027406
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    # Data series
    simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
    # TODO : remove whwere geoglows server works
    simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()

    simulated_data = simulated_data.rename(columns={
                                "streamflow_m^3/s": "Historical simulation (m3/s)"})
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    simulated_data.to_excel(writer, sheet_name='serie_historica_simulada', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=serie_historica_simulada.xlsx'
    response.write(output.getvalue())
    return response


# Retrieve xlsx data
@controller(name='get_forecast_xlsx',url='hydroviewer-colombia/get-forecast-xlsx')
def get_forecast_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid']
    forecast_date = request.GET['fecha']
    
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    try:
        forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    finally:
        conn.close()
    
    # Raw forecast
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    ensemble_stats.to_excel(writer, sheet_name='ensemble_stats', index=True)  # Aquí se incluye el índice
    ensemble_forecast.to_excel(writer, sheet_name='ensemble_forecast', index=True)  # Aquí se incluye el índice
    forecast_records.to_excel(writer, sheet_name='forecast_records', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=ensemble_forecast.xlsx'
    response.write(output.getvalue())
    return response


# Retrieve xlsx data
@controller(name='get_fews_streamflow_data_xlsx',url='hydroviewer-colombia/get-fews-streamflow-data-xlsx')
def get_fews_streamflow_data_xlsx(request):
    # Retrieving GET arguments
    station_code = request.GET['code']

    # Database download
    obs_st, sen_st = get_fews_data(station_code, 'jsonQ', 'Qobs')
    rv = pd.merge(left=obs_st.reset_index(),
                  right=sen_st.reset_index(),
                  how='outer', 
                  on='date',
                  suffixes=['Caudal observado', 'Caudal sensor'],
                  )
    rv.rename(columns={col : col.replace('data', '') for col in rv.columns}, inplace=True)
    rv.sort_values(by=['date'], inplace=True)
    rv.reset_index(drop=True, inplace=True)

    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    rv.to_excel(writer, sheet_name='Caudal tiempo real', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)

    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=caudal_tiempo_real_{}.xlsx'.format(station_code)
    response.write(output.getvalue())
    return response


# Retrieve xlsx data
@controller(name='get_fews_water_level_data_xlsx',url='hydroviewer-colombia/get-fews-water-level-data-xlsx')
def get_fews_water_level_data_xlsx(request):
    # Retrieving GET arguments
    station_code = request.GET['code']

    # Database download
    obs_wl, sen_wl = get_fews_data(station_code, 'jsonH', 'Hobs')
    rv = pd.merge(left=obs_wl.reset_index(),
                  right=sen_wl.reset_index(),
                  how='outer', 
                  on='date',
                  suffixes=['Nivel observado', 'Nivel sensor'],
                  )
    rv.rename(columns={col : col.replace('data', '') for col in rv.columns}, inplace=True)
    rv.sort_values(by=['date'], inplace=True)
    rv.reset_index(drop=True, inplace=True)

    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    rv.to_excel(writer, sheet_name='Nivel tiempo real', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)

    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=nivel_tiempo_real_{}.xlsx'.format(station_code)
    response.write(output.getvalue())
    return response


############################################################
@controller(url  = "hydroviewer-colombia/user-manual",
            name = "user_manual")
def user_manual(request):
    context = {}
    return render(request, 'hydroviewer_colombia/user_manual.html', context)


@controller(url = "hydroviewer-colombia/technical-manual",
            name = "technical_manual")
def technical_manual(request):
    context = {}
    return render(request, 'hydroviewer_colombia/technical_manual.html', context)

############################################################
#                          SERVICES                        #
############################################################
@api_view(['GET'])
@authentication_classes((TokenAuthentication,))
@controller(name='get_image',
            url='hydroviewer-colombia/get-image',
            login_required=False)
def down_load_img(request):
    
    # Retrieving GET arguments
    station_comid  = request.GET['comid']
    type_graph = request.GET['typeGraph']

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()

    try:
        # Data series
        simulated_data = get_format_data("select * from hs_{0};".format(station_comid), conn)
        # TODO : remove whwere geoglows server works
        simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()

        ensemble_forecast = get_format_data("select * from f_{0};".format(station_comid), conn)
        forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    finally:
        conn.close()

    db.dispose()

    # Calc ensemble and return periods
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    return_periods = get_return_periods(station_comid, simulated_data)

    # Plots data
    if 'historical' == type_graph:
        """
        fig = geoglows.plots.historic_simulation(
                                        hist = simulated_data,
                                        rperiods = return_periods,
                                        outformat = "plotly",
                                        titles = {'Reach COMID': station_comid})
        """
        fig = historic_simulation(
                                hist = simulated_data,
                                rperiods = return_periods,
                                outformat = "plotly",
                                titles = {'COMID': station_comid})
        name_file = 'historical'
    elif 'forecast' == type_graph:
        fig = get_forecast_plot(
                                comid = station_comid, 
                                stats = ensemble_stats, 
                                rperiods = return_periods, 
                                records = forecast_records)
        name_file = 'corrected_forecast'

    # Build image bytes
    img_bytes = fig.to_image(format="png")

    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type="image/jpeg")
    response['Content-Disposition'] = 'attachment; filename={0}_{1}_RQ.png'.format(name_file, station_comid)
    response.write(img_bytes)

    return response