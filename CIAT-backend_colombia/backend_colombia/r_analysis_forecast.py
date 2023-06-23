import os
import sys
import datetime
import numpy as np
import pandas as pd
import geoglows as ggs
from dotenv import load_dotenv
from sqlalchemy import create_engine

from backend_auxiliar import gumbel_1

#################################################################
#                                                               #
#                  r_analysis_forecast.py                       #
#                                                               #
#################################################################

class Update_alarm_level:
	def __init__(self, pgres_tab_obshist, pgres_tab_name):

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

		# Postgres secure data
		pgres_password     = DB_PASS
		pgres_databasename = DB_NAME

		# Database to update
		# pgres_tab_name        = 'stations_streamflow'
		pgres_tab_coln_id     = 'codigo'
		pgres_tab_coln_comid  = 'comid'
		pgres_tab_coln_update = 'alert'

		# Database
		pgres_tab_dname_funct = lambda x : 's_'.format(x)
		# self.pgres_tab_obshist = 'observed_streamflow_data'
		self.pgres_tab_obshist = pgres_tab_obshist
		self.pgres_tab_obshist_func = lambda x : 's_{}'.format(x)
		self.pgres_tab_simhist_func = lambda x : 'hs_{}'.format(x)
		self.pgres_tab_forecst_func = lambda x : 'f_{}'.format(x)

		self.return_periods = [100, 50, 25, 10, 5, 2]

		self.accepted_warning = 10

		######################### MAIN ########################
		# Establish connection
		db   = create_engine("postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER,
																					   pgres_password,
																					   pgres_databasename))

		# Load data to transform
		conn = db.connect()
		try:
			df_id_alert = pd.read_sql('select {} , {} , {} from {}'.format(pgres_tab_coln_id,
																	  pgres_tab_coln_comid,
																	  pgres_tab_coln_update,
																      pgres_tab_name), 
								      con=conn,
									  )
		finally:
			conn.close()

		# Fix pgres_tab_coln_update
		df_id_alert.rename(columns={pgres_tab_coln_update : 'old'}, inplace=True)

		# Build alert to update
		alert_rv = []
		for num, row in df_id_alert.iterrows():
			
			row_id    = row[pgres_tab_coln_id]

			row_comid = row[pgres_tab_coln_comid]
			row_data  = row['old']

			# Load data to calc the alert level
			conn = db.connect()
			try:
				obs_hist_df = self.load_observed_historical_data(row_id, conn)
				sim_hist_df = self.load_simulated_historical_data(row_comid, conn)
				forecast_df = self.load_forecast_data(row_comid, conn)
			finally:
				conn.close()		

			# Fix column index
			forecast_df.set_index('datetime', inplace=True)
			forecast_df.index = pd.to_datetime(forecast_df.index, '%Y-%m-%d %H:%M:%S')

			sim_hist_df.set_index('datetime', inplace=True)
			sim_hist_df.index = pd.to_datetime(sim_hist_df.index, '%Y-%m-%d')
			sim_hist_df.where(sim_hist_df > 0, 0, inplace=True)

			obs_hist_df.set_index('datetime', inplace=True)
			obs_hist_df.index = pd.to_datetime(obs_hist_df.index, '%Y-%m-%d')

			# Fix simulated and observed data at same dates
			sim_hist_df, obs_hist_df = self.__asincro_df__(sim_hist_df, obs_hist_df)

			
			if len(obs_hist_df.index.month.unique()) < 12:
				print(15 * '-')
				print('Problem with station')
				print(row_id)
				print(15 * '-')

			# Bias correction fix data
			forecast_df = self.__bias_correction_forecast__(fore_nofix = forecast_df,
														    sim_hist   = sim_hist_df,
															obs_hist   = obs_hist_df)
			sim_hist_df = self.__bias_correction__(sim_hist_df, obs_hist_df)

			# Calc warnings level
			warnings_level = self.__get_warning_level__(comid = row_id,
												        data  = sim_hist_df)

			# Forecast stats
			ensemble_stats = self.get_ensemble_stats(forecast_df)

			# Obtein alert
			alert = self.get_excced_rp(ensemble_stats, forecast_df, warnings_level)

			# Result
			alert_rv.append(alert)
			
			if alert != 'R0':
				print(row_id, alert)

		# """
		conn = db.connect()
		try:
			df = pd.read_sql('select * from {}'.format(pgres_tab_name), 
						      con=conn)
			df[pgres_tab_coln_update] = alert_rv
			df.to_sql(pgres_tab_name, conn, if_exists='replace', index=False)
		finally:
			conn.close()
		# """


	def __asincro_df__(self, sim, obs):
		
		sim.reset_index(inplace = True)
		obs.reset_index(inplace = True)

		rv = pd.merge(sim, obs, on='datetime', how='outer')
		rv.dropna(inplace=True)

		sim = rv[sim.columns]
		obs = rv[sim.columns]

		sim.set_index('datetime', inplace = True)
		obs.set_index('datetime', inplace = True)
		
		return sim, obs


	# Excedence warning
	def get_excced_rp(self, stats: pd.DataFrame, ensem: pd.DataFrame, rperiods: pd.DataFrame):
		dates = stats.index.tolist()
		startdate = dates[0]
		enddate = dates[-1]
		span = enddate - startdate
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
		if(self.__is_warning__(r2)):
			alarm = "R2"
		if(self.__is_warning__(r5)):
			alarm = "R5"
		if(self.__is_warning__(r10)):
			alarm = "R10"
		if(self.__is_warning__(r25)):
			alarm = "R25"
		if(self.__is_warning__(r50)):
			alarm = "R50"
		if(self.__is_warning__(r100)):
			alarm = "R100"
		#out = pd.DataFrame({"rp2": r2, "rp5": r5, "rp10": r10, "rp25": r25, "rp50": r50, "rp100": r100})
		return(alarm)


	def __is_warning__(self, arr):
		cond = [i >= self.accepted_warning for i in arr].count(True) > 0
		return(cond)


	# Ensemble methods
	def get_ensemble_stats(self, ensemble):
		high_res_df = ensemble['ensemble_52_m^3/s'].to_frame()
		ensemble.drop(columns=['ensemble_52_m^3/s'], inplace=True)
		ensemble.dropna(inplace= True)
		high_res_df.dropna(inplace= True)
		high_res_df.rename(columns = {'ensemble_52_m^3/s':'high_res_m^3/s'}, inplace = True)
		stats_df = pd.concat([
			self.ensemble_quantile(ensemble, 1.00, 'flow_max_m^3/s'),
			self.ensemble_quantile(ensemble, 0.75, 'flow_75%_m^3/s'),
			self.ensemble_quantile(ensemble, 0.50, 'flow_avg_m^3/s'),
			self.ensemble_quantile(ensemble, 0.25, 'flow_25%_m^3/s'),
			self.ensemble_quantile(ensemble, 0.00, 'flow_min_m^3/s'),
			high_res_df
							], axis=1)
		return stats_df


	def ensemble_quantile(self, ensemble, quantile, label):
		df = ensemble.quantile(quantile, axis=1).to_frame()
		df.rename(columns = {quantile: label}, inplace = True)
		return df


	# Warning calc methods
	def __get_warning_level__(self, comid, data):
		# Stats
		max_annual_flow = data.groupby(data.index.year).max()
		mean_value      = np.mean(max_annual_flow.iloc[:,0].values)
		std_value       = np.std(max_annual_flow.iloc[:,0].values)

		# Return periods
		return_periods_values = []

		# TODO : Remove the for, function take a lot of time..
		# Compute the corrected return periods
		for rp in self.return_periods:
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

		return corrected_rperiods_df
	

	# Bias correction methods
	def __bias_correction_forecast__(self, sim_hist, fore_nofix, obs_hist):
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
		corrected_ensembles = ggs.bias.correct_forecast(forecast_ens_df, sim_hist, obs_hist)
		corrected_ensembles = corrected_ensembles.multiply(min_factor_df, axis=0)
		corrected_ensembles = corrected_ensembles.multiply(max_factor_df, axis=0)

		return corrected_ensembles


	def __bias_correction__(self, sim_hist, obs_hist):
		return ggs.bias.correct_historical(simulated_data = sim_hist,
										   observed_data  = obs_hist)


	# Methods for load data
	def load_observed_historical_data(self, id_data, conn):
		# Load observed historical data form postgres
		return pd.read_sql('select {} , datetime from {}'.format(self.pgres_tab_obshist_func(id_data), 
																 self.pgres_tab_obshist), con=conn)


	def load_simulated_historical_data(self, comid_data, conn):
		# Load historical simulated data from postgres
		return pd.read_sql('select * from {}'.format(self.pgres_tab_simhist_func(comid_data)), con=conn)


	def load_forecast_data(self, comid_data, conn):
		# Load forecast data
		return pd.read_sql('select * from {}'.format(self.pgres_tab_forecst_func(comid_data)), con=conn)


if __name__ == "__main__":
	# Update alarm level
	
	print(' {} '.center(70, '-').format(datetime.date.today()))

	print('Stream flow data update')
	pgres_tab_obshist = 'observed_streamflow_data'
	pgres_tab_name = 'stations_streamflow'
	Update_alarm_level(pgres_tab_obshist, pgres_tab_name)
	
	print('Water level data update')
	pgres_tab_obshist = 'observed_waterlevel_data'
	pgres_tab_name = 'stations_waterlevel'
	Update_alarm_level(pgres_tab_obshist, pgres_tab_name)
