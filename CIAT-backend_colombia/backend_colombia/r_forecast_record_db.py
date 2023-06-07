import os
import sys
from dotenv import load_dotenv
import io
import numpy as np
import pandas as pd
import datetime as dt
import concurrent.futures
from sqlalchemy import create_engine

import time

from backend_auxiliar import data_request
import sys

####################################################################
#                                                                  #
#                     r_forecast_record_db.py                      #
#                                                                  #
####################################################################


class Update_forecast_record_db:
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
		pgres_password       = DB_PASS
		pgres_databasename   = DB_NAME
		self.pgres_tablename_func = lambda comid : 'fr_{}'.format(comid)

		# Comid column name from postgres database
		# TODO : Only for test
		station_table_name = 'drainage'
		# station_table_name = 'stations_streamflow'
		station_comid_name = 'HydroID'
		# station_comid_name = 'comid'

		# GEOGloWS Streamflow Servises dictionary
		url     = 'https://geoglows.ecmwf.int/api/ForecastRecords/' 
		url_fun = lambda x : (url ,  {'reach_id'      : x,
				                      'return_format' : 'csv'}) 
		self.dict_aux = {'Datetime column name'    : 'datetime',
						'Datetime column format'  : '%Y-%m-%dT%H:%M:%SZ',
						'Data column name'        : 'streamflow_m^3/s',
						'Data column name prefix' : 'c_',
						'Days to download'        : 7, # 0 or neg value -> Download all data
						}

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
		# comids = comids[:20]

		# Build start date
		start_date = dt.date.today() - dt.timedelta(days=self.dict_aux['Days to download']) 
		start_date = start_date.strftime('%Y%m%d')

		# Split list for clear the cache
		comids_chunk = np.array_split(comids, n_chunks)

		# Run chunk by chunk
		for chunk, comids in enumerate(comids_chunk, start = 1):

			# Download data
			with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
				list(executor.map(lambda c : self.__parallelization__(c, url_fun, start_date, db),
								  comids)
					)

			print('Update : {:.0f} %, Delay : {:.4f} seg.'.format(100 * chunk / n_chunks, time.time() - before))


	def __parallelization__(self, c, url_fun, start_date, db):
		session = db.connect()
		try:
		 	self.__download_data__(c, url_fun, start_date, session)
		finally:
		 	session.close()

	
	def __download_data__(self, 
						  comid : str, 
						  url : "func",  
						  start_date: str,
						  db : "POSTGRES database"):
		"""
		Seriealized download function
		Input:
			comid      : str  -> comid to download
			url        : func -> function to download data
			start_date : str  -> Date to start the data
			db         : pgdb -> Postgres database
		"""
		# print('Downloding : {}'.format(comid))
	
		# Get data for download
		url_comid, params_comid = url(comid)

		# Fix days to download
		if self.dict_aux['Days to download'] > 0:
			params_comid['start_date'] = start_date #YYYYMMDD
		
		# Make server request
		df = data_request(url=url_comid, params=params_comid)
		df = self.__build_dataframe__(df, url_comid, params_comid)

		# Review of number of data
		if df.shape[0] <= 2:
			df = data_request(url=url_comid, params=params_comid)
			df = self.__build_dataframe__(df, url_comid, params_comid)

		# Fix column names
		df.rename(columns = {self.dict_aux['Data column name'] : \
							 self.dict_aux['Data column name prefix'] + str(comid)},
					   inplace = True)

		# Insert to database
		df.to_sql(self.pgres_tablename_func(comid), con=db, if_exists='replace', index=True)

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
			The next try-except block has the same function and the same result. In the try statement, the result 
			is fast but requires a correct date format. In the except statement, the date format is not needed, but 
			it is slower.
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
	print('Updating forecast record - {}'.center(70, '-').format(dt.date.today()))
	Update_forecast_record_db()
	print('Updated forecast record'.center(70, '-'))

