####################################################################################################
##                                   LIBRARIES AND DEPENDENCIES                                   ##
####################################################################################################

# Tethys platform
from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from tethys_sdk.routing import controller
from tethys_sdk.gizmos import PlotlyView, DatePicker

# Postgresql
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from pandas_geojson import to_geojson

# Geoglows
import geoglows
import numpy as np
import math
import hydrostats as hs
import hydrostats.data as hd
import HydroErr as he
import plotly.graph_objs as go
import datetime as dt

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



####################################################################################################
##                                      PLOTTING FUNCTIONS                                        ##
####################################################################################################

# Plotting daily averages values
def get_daily_average_plot(sim, comid):
    # Generate the average values
    daily_avg_sim = hd.daily_average(sim)
    # Generate the plots on Ploty
    daily_avg_sim_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 0].values, name = 'Simulated', )
    # PLot Layout
    layout = go.Layout(
        title='Daily average streamflow <br>Reach COMID: {0}'.format(comid),
        xaxis=dict(title='Days', ), 
        yaxis=dict(title='Discharge (m<sup>3</sup>/s)', autorange=True),
        showlegend=False)
    # Generate the output
    chart_obj = go.Figure(data=[daily_avg_sim_Q], layout=layout)
    return(chart_obj)

# Plotting monthly averages values
def get_monthly_average_plot(sim, comid):
    # Generate the average values
    daily_avg_sim = hd.monthly_average(sim)
    # Generate the plots on Ploty
    daily_avg_sim_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 0].values, name = 'Simulated', )
    # PLot Layout
    layout = go.Layout(
        title='Monthly Average Streamflow <br>Reach COMID: {0}'.format(comid),
        xaxis=dict(title='Months', ), 
        yaxis=dict(title='Discharge (m<sup>3</sup>/s)', autorange=True),
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
    simulated_volume = go.Scatter(x = sim.index, y = sim_volume, name='Simulated', )
    # Plot layouts
    layout = go.Layout(
                title='Simulated Cumulative Volume <br>{0}'.format(comid),
                xaxis=dict(title='Dates', ), 
                yaxis=dict(title='Volume (Mm<sup>3</sup>)', autorange=True),
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
    hydroviewer_figure = geoglows.plots.forecast_stats(stats=corrected_stats_df, titles={'Reach ID': comid})
    x_vals = (corrected_stats_df.index[0], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[len(corrected_stats_df.index) - 1], corrected_stats_df.index[0])
    max_visible = max(corrected_stats_df.max())
    ##
    corrected_records_plot = fixed_records.loc[fixed_records.index >= pd.to_datetime(corrected_stats_df.index[0] - dt.timedelta(days=8))]
    corrected_records_plot = corrected_records_plot.loc[corrected_records_plot.index <= pd.to_datetime(corrected_stats_df.index[0] + dt.timedelta(days=2))]
    ##
    if len(corrected_records_plot.index) > 0:
      hydroviewer_figure.add_trace(go.Scatter(
          name='1st days forecasts',
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
    hydroviewer_figure.add_trace(template('Return Periods', (r100 * 0.05, r100 * 0.05, r100 * 0.05, r100 * 0.05), 'rgba(0,0,0,0)', fill='none'))
    hydroviewer_figure.add_trace(template(f'2 Year: {r2}', (r2, r2, r5, r5), colors['2 Year']))
    hydroviewer_figure.add_trace(template(f'5 Year: {r5}', (r5, r5, r10, r10), colors['5 Year']))
    hydroviewer_figure.add_trace(template(f'10 Year: {r10}', (r10, r10, r25, r25), colors['10 Year']))
    hydroviewer_figure.add_trace(template(f'25 Year: {r25}', (r25, r25, r50, r50), colors['25 Year']))
    hydroviewer_figure.add_trace(template(f'50 Year: {r50}', (r50, r50, r100, r100), colors['50 Year']))
    hydroviewer_figure.add_trace(template(f'100 Year: {r100}', (r100, r100, max(r100 + r100 * 0.05, max_visible), max(r100 + r100 * 0.05, max_visible)), colors['100 Year']))
    ##
    hydroviewer_figure['layout']['xaxis'].update(autorange=True)
    return(hydroviewer_figure)


####################################################################################################
##                                   CONTROLLERS AND REST APIs                                    ##
####################################################################################################

# Initialize the web app
@controller(name='home', url='hydroviewer-colombia')
def home(request):
    return render(request, 'hydroviewer_colombia/home.html')


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
    ensemble_forecast = get_format_data("select * from f_{0};".format(station_comid), conn)
    forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    return_periods = get_return_periods(station_comid, simulated_data)

    # Close conection
    conn.close()

    # Plots and tables
    data_plot = geoglows.plots.historic_simulation(
                                    hist = simulated_data,
                                    rperiods = return_periods,
                                    outformat = "plotly",
                                    titles = {'Reach COMID': station_comid})
    daily_average_plot = get_daily_average_plot(
                                    sim = simulated_data,
                                    comid = station_comid)
    monthly_average_plot = get_monthly_average_plot(
                                    sim = simulated_data,
                                    comid = station_comid)
    flow_duration_curve = geoglows.plots.flow_duration_curve(
                                    hist = simulated_data,
                                    outformat = "plotly",
                                    titles = {'Reach COMID': station_comid})
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
    # Raw forecast
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    ensemble_stats.to_excel(writer, sheet_name='ensemble_forecast', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=ensemble_forecast.xlsx'
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