# Import libraries and dependencies
import os
import math
import psycopg2
import geoglows
import datetime
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')

# Change the work directory
user = os.getlogin()
user_dir = os.path.expanduser('~{}'.format(user))
os.chdir(user_dir)

try:
    os.chdir("tethys-apps-colombia/CIAT-backend_colombia/backend_colombia/")
except:
    os.chdir("/home/jrc/colombia-tethys-apps/CIAT-backend_colombia/backend_colombia/")

# Import enviromental variables
load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME = os.getenv('DB_NAME')

# Generate the conection token
token = "postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER, DB_PASS, DB_NAME)


###############################################################################################################
#                                 Function to get and format the data from DB                                 #
###############################################################################################################
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
    return data



###############################################################################################################
#                                   Getting return periods from data series                                   #
###############################################################################################################
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
    
    
    # JRC FIX
    # return_periods_values = [gumbel_1(std_value, mean_value, rp) for rp in return_periods]
    
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
    return corrected_rperiods_df


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

    low_warnings_number_id = {'7q10' : 1}

    # Previous datatime manager
    data_cp = data.copy()
    data_cp = data_cp.rolling(window=7).mean()
    data_cp = data_cp.groupby(data_cp.index.year).min().values.flatten()

    # Calc comparation value
    rv = {}
    for key in low_warnings_number_id:
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

###############################################################################################################
#                                         Getting ensemble statistic                                          #
###############################################################################################################
def ensemble_quantile(ensemble, quantile, label):
    df = ensemble.quantile(quantile, axis=1).to_frame()
    df.rename(columns = {quantile: label}, inplace = True)
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
        high_res_df
    ], axis=1)
    return stats_df


###############################################################################################################
#                                    Warning if exceed x return period                                        #
###############################################################################################################
def is_warning(arr):
    cond = [i >= 40 for i in arr].count(True) > 0
    return cond

def get_excced_rp(stats: pd.DataFrame, ensem: pd.DataFrame, rperiods: pd.DataFrame):
    dates     = stats.index.tolist()
    startdate = dates[0]
    enddate   = dates[-1]
    span      = enddate - startdate
    uniqueday = [startdate + datetime.timedelta(days=i) for i in range(span.days + 2)]
    
    # get the return periods for the stream reach
    rp2 = rperiods['return_period_2'].values
    rp5 = rperiods['return_period_5'].values
    rp10 = rperiods['return_period_10'].values
    rp25 = rperiods['return_period_25'].values
    rp50 = rperiods['return_period_50'].values
    rp100 = rperiods['return_period_100'].values

    # fill the lists of things used as context in rendering the template
    days = []
    r2 = []
    r5 = []
    r10 = []
    r25 = []
    r50 = []
    r100 = []
    
    for i in range(len(uniqueday) - 1):  # (-1) omit the extra day used for reference only
        tmp = ensem.loc[uniqueday[i]:uniqueday[i + 1]]
        days.append(uniqueday[i].strftime('%b %d'))
        num2 = 0
        num5 = 0
        num10 = 0
        num25 = 0
        num50 = 0
        num100 = 0
        for column in tmp:
            column_max = tmp[column].to_numpy().max()
            if column_max > rp100:
                num100 += 1
            if column_max > rp50:
                num50 += 1
            if column_max > rp25:
                num25 += 1
            if column_max > rp10:
                num10 += 1
            if column_max > rp5:
                num5 += 1
            if column_max > rp2:
                num2 += 1
        r2.append(round(num2 * 100 / 52))
        r5.append(round(num5 * 100 / 52))
        r10.append(round(num10 * 100 / 52))
        r25.append(round(num25 * 100 / 52))
        r50.append(round(num50 * 100 / 52))
        r100.append(round(num100 * 100 / 52))
        
    alarm = "R0"
    if(is_warning(r2)):
        alarm = "R2"
    if(is_warning(r5)):
        alarm = "R5"
    if(is_warning(r10)):
        alarm = "R10"
    if(is_warning(r25)):
        alarm = "R25"
    if(is_warning(r50)):
        alarm = "R50"
    if(is_warning(r100)):
        alarm = "R100"
    
    return alarm


# Excedence warning low
def get_occurrence_low_warning(ensem: pd.DataFrame, warnings: pd.DataFrame):

    # Build esnsemble comparation time serie
    ts = ensem.median(axis = 1).copy()
    ts = ts.groupby(ts.index.year.astype(str) +'/'+ ts.index.month.astype(str) +'/'+ ts.index.day.astype(str)).min()

    # Count warnings alerts
    rv = {}
    for warning in warnings.columns:
        rv[warning] = len(ts[ts < warnings[warning].values[0]])

    # Assing warnings
    if rv['7q10'] >= 3 and rv['7q10'] < 7 :
        return 'lower_1'
    elif rv['7q10'] >= 7 and rv['7q10'] < 10 :
        return 'lower_3'
    elif rv['7q10'] >= 10 :
        return 'lower_7'
    else:
        return 'R0'



#######################################################################
#                                   MAIN
#######################################################################

# Setting the connetion to db
db = create_engine(token)
try:
    # Establish connection
    conn = db.connect()
    try:
        # Getting stations
        drainage = pd.read_sql("select * from drainage", conn)
    finally:
        # Close connection
        conn.close()
finally:
    db.dispose()

print(' {} '.center(70, '-').format(datetime.date.today()))

# Number of stations
# n = len(drainage)
comids = drainage['hydroid'].values.tolist()
n      = len(comids)
new_alert = []

# Extract alert
db = create_engine(token)

try:
    for i, station_comid in enumerate(comids):
        
        # Query to database
        conn = db.connect()
        try:
            # Establish connection
            simulated_data = get_format_data("select * from hs_{0}".format(station_comid), conn)
            # TODO : remove whwere geoglows server works
            simulated_data = simulated_data[simulated_data.index < '2022-06-01'].copy()

            ensemble_forecast = get_format_data("select * from f_{0}".format(station_comid), conn)
        finally:
            # Close connection
            conn.close()
        
        # Return period
        return_periods = get_return_periods(station_comid, simulated_data)
        warnings_low_level = get_warning_low_level(comid = station_comid,
							    				   data  = simulated_data)
        
        
        del simulated_data

        # Forecast stats
        ensemble_stats = get_ensemble_stats(ensemble_forecast)
        
        # Warning if excced a given return period in 10% of emsemble
        alert_val = get_excced_rp(ensemble_stats, ensemble_forecast, return_periods)

        if alert_val == 'R0':
            alert_val = get_occurrence_low_warning(ensemble_stats, warnings_low_level)

        new_alert.append(alert_val)
        

        if alert_val != 'R0':
            print("Progress: {0} %. Comid: {1}. Alert : {2}".format(round(100 * i/n, 3), station_comid, alert_val))
finally:
    db.dispose()


# Update data
drainage['alert'] = new_alert

# Save data
db = create_engine(token)
try:
    # Establish connection
    conn = db.connect()

    # Insert to database
    try:
        drainage.to_sql('drainage', con=conn, if_exists='replace', index=False)

    finally:
        # Close connection
        conn.close()

finally:
    db.dispose()
