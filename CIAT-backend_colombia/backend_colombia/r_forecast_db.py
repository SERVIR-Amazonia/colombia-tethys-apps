import os
from dotenv import load_dotenv
import io
import sys
# import requests
import numpy as np
import pandas as pd
import datetime as dt
# from tqdm import tqdm
import concurrent.futures
from sqlalchemy import create_engine

import time

from backend_auxiliar import data_request

####################################################################
#                                                                  #
#                         r_forecast_db.py                         #
#                                                                  #
####################################################################


class Update_forecast_db:
	def __init__(self):

		before = time.time()
		n_chunks = 10

		# Change the work directory
		user = os.getlogin()
		user_dir = os.path.expanduser('~{}'.format(user))
		os.chdir(user_dir)

		try:
			os.chdir("tethys_apps_colombia/CIAT-backend_colombia/backend_colombia/")
		except:
			os.chdir("/home/jrc/CIAT-backend_colombia/backend_colombia/")


		# Import enviromental variables
		load_dotenv()
		DB_USER = os.getenv('DB_USER')
		DB_PASS = os.getenv('DB_PASS')
		DB_NAME = os.getenv('DB_NAME')

		# Postgres secure data
		pgres_password     = DB_PASS
		pgres_databasename = DB_NAME
		pgres_tablename    = 'drainage'

		# Postgres secure data
		pgres_password       = DB_PASS
		pgres_databasename   = DB_NAME
		self.pgres_tablename_func = lambda comid : 'f_{}'.format(comid)

		# Comid column name from postgres database
		# TODO : Only for test
		station_table_name = 'drainage'
		# station_table_name = 'stations_streamflow'
		station_comid_name = 'HydroID'
		# station_comid_name = 'comid'

		# GEOGloWS Streamflow Servises dictionary
		url     = 'https://geoglows.ecmwf.int/api/ForecastEnsembles/'
		url_fun = lambda x : (url ,  {'reach_id'      : x,
				                      'return_format' : 'csv'}) 
		self.dict_aux = {'Datetime column name'   : 'datetime',
						 'Datetime column format' : '%Y-%m-%dT%H:%M:%SZ'}

		# ------------------- MAIN --------------------
		# Establish connection
		db   = create_engine("postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER,
																					   pgres_password, 
 																					   pgres_databasename))
	
		# Read comid list
		conn = db.connect()
		try:
			comids = pd.read_sql('select {} from {}'.format(station_comid_name, station_table_name), conn)\
					   .values\
					   .flatten()\
					   .tolist()
		finally:
			conn.close()

		# In case of one comid is requiered, only remove the comment simbol (#) and in the list add the
		# comid to call
		# comids = comids[:100]

		# Split list for clear the cache memory
		comids_chunk = np.array_split(comids, n_chunks)

		# Run chunk by chunk
		for chunk, comids in enumerate(comids_chunk, start = 1):

			# Download data from comid
			with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
				_ = list(executor.map(lambda c : self.__parallelization__(c, url_fun, db), 
									  comids))

			print('Update : {:.0f} %, Delay : {:.4f} seg'.format(100 * chunk / n_chunks, time.time() - before))


	def __parallelization__(self, c, url_fun, db):
		# Parallelization of ssesion out of daownload data
		session = db.connect()
		try:
			self.__download_data__(c, url_fun, session)
		finally:
			session.close()


	def __download_data__(self, comid, url, db):
		"""
		Seriealized download function
		Input:
			comid      : str   -> comid to download
			url        : func  -> function to download data
			db         : pgres -> Postgres database
		Output :
			str        : str  -> String in bytes downloaded
		"""
		# Get data for download
		url_comid, params_comid = url(comid)

		# Make server request
		df = data_request(url=url_comid, params=params_comid)
		df = self.__build_dataframe__(df, url_comid, params_comid)

		# Review number of data download
		if df.shape[1] != 52 or df.shape[0] <= 2:
			df = data_request(url=url_comid, params=params_comid)
			df = self.__build_dataframe__(df, url, params)

		# Build table name
		table_name = self.pgres_tablename_func(comid)
		
		# Insert to data
		df.to_sql(table_name, con=db, if_exists='replace', index=True)

		# print('Download : {}'.format(comid))

		return 0


	def __build_dataframe__(self, input_data, url, params):
		"""
		Build dataframe from return value of data_request sunction.
		Input :
			input_data : str/bites -> Return of data_request function
		Output:
			pandas.DataFrame -> Table with results
		"""
		if "ERROR" == input_data:
			rv = pd.DataFrame(data = {self.dict_aux['Datetime column name']    : [pd.NaT],
									  self.dict_aux['Data column name prefix'] : [float('nan')]})
			return rv
		else:
			"""
			The next try-except block has the same function and the same result. In the try statement, 
			the result is fast but requires a correct date format. In the except statement, the date
			format is not needed, but it is slower.
			"""
			try:
				rv = pd.read_csv(io.StringIO(input_data),
				 				 parse_dates = [self.dict_aux['Datetime column name']],
								 date_parser = lambda x : dt.datetime.strptime(x,
								 											   self.dict_aux['Datetime column format']),
								 index_col   = [self.dict_aux['Datetime column name']],
								)
			except Exception as e:
				# TODO : Try to remove this pice of code or change location
				# If the data download fails, the download process will be recursive.
				print('Exception: {}'.format(e))
				df = data_request(url, params)
				rv = self.__build_dataframe__(df, url, params)

			finally:
				return rv


if __name__ == "__main__":
	print(' Updating forecast - {} '.center(70, '-').format(dt.date.today()))
	Update_forecast_db()
	print(' Updated forecast '.center(70, '-'))

