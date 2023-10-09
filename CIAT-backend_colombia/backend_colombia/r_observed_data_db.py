import os
import io
import sys
import requests
import psycopg2
import pandas as pd
import datetime as dt
from dotenv import load_dotenv
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


#############################################################
#                                                           #
#                   r_observed_data.py                      #
#                                                           #
#############################################################


class Update_historical_observed_data:
	def __init__(self,  pgres_table_name, 
						station_table_name,
						station_id_name,
						url_hs,
						id_HS_stations,
						func_url_build,
						dict_names):

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

		# Postgres user data pass
		pgres_password     = DB_PASS
		pgres_databasename = DB_NAME

		# URL from data in hydroshare
		self.dict_names = dict_names

		# --------------- MAIN ------------------
		# Establish connection
		db   = create_engine("postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER,
																					   pgres_password, 
																					   pgres_databasename))

		# Connection
		conn = db.connect()
		try:
			# Read station list
			stations = pd.read_sql('select {} from {}'.format(station_id_name.lower(), station_table_name), conn)\
						.values\
						.flatten()\
						.tolist()
		finally:
			conn.close()

		# Read data of stations
		df = self.__data_from_hydroshare__(stations = stations,
										   func_url = func_url_build)

		# Make connection
		conn = db.connect()
		
		try:
			# Insert to database
			df.to_sql(pgres_table_name, con=conn, if_exists='replace', index=True)
		finally:
			# Close connection
			conn.close()


	def __data_from_hydroshare__(self, 
								 stations : list, 
								 func_url : "Function",
								 ) -> pd.DataFrame:
		"""
		Download and sort the data form hydroshare.
		Input : 
			stations   : list       -> List of the stations
			func_url   : "function" -> Function with url construccion from station name/code
		Output :
			pandas.DataFrame : Couplling data for all stations
		"""

		# TODO : Paralellization is possible, but the station list need to be splited and others.
		# Time tested : 138 s
		# before = time.time()
		for station in stations:
			station_data = self.__download_from_comid__(station, func_url)
			try:
				rv = station_data.merge(rv, 
										on  = self.dict_names['Datetime column name'].lower(), 
										how = 'outer')
			except:
				rv = station_data.copy()
		# print('delay in sec: {:.2f}'.format(time.time() - before))

		# Fix index name
		rv.sort_values(by = self.dict_names['Datetime column name'].lower(), inplace=True)
		rv.set_index(self.dict_names['Datetime column name'].lower(), 
                     inplace = True)

		return rv


	def __download_from_comid__(self, id_name, func_url, cnt_fail = 0):
		"""
		Download the data for hydrohare
		Input : 
			id : str        -> the id of the station.
		Output :
			pd.DataFrame    -> Data of station
		"""
		
		data = requests.get(func_url(id_name), verify=False)

		if data.status_code == 200:
			# Success condition
			rv = data.content
			data.close()

			df = pd.read_csv(io.StringIO(rv.decode('utf-8')),
							 parse_dates = [self.dict_names['Datetime column name']],
							 date_parser = lambda x : dt.datetime.strptime(x, 
													  self.dict_names['Datetime column format']))

			df.rename(columns = {self.dict_names['Datetime column name'] : \
								 self.dict_names['Datetime column name'].lower(),
								 self.dict_names['Data column name'] : \
								 self.dict_names['Data column name prefix'] + str(id_name)},
					  inplace = True)



			print('Download : {}'.format(id_name))
			return df

		elif data.status_code != 200 and cnt_fail > 5:
			# Failure condition
			data.close()
			print('Error in download station {}'.format(id_name))
			return pd.DataFrame(data = {self.dict_names['Datetime column name'].lower()           : [pd.NaT],
										self.dict_names['Data column name prefix'] + str(id_name) : [float('nan')]})

		else:
			# Restart condition
			data.close()
			cnt_fail += 1
			return self.__download_from_comid__(id_name=id_name,
												func_url=func_url,
												cnt_fail = cnt_fail)


if __name__ == '__main__':
	# TODO : Add download observed data for water level forecast
	print(' Data database updating. '.center(70, '-'))

	# General data
	url_hs             = 'https://www.hydroshare.org/resource/'


	# """
	print(' Streamflow '.center(70, '-'))
	# For streamflow download
	station_table_name = 'stations_streamflow'
	pgres_table_name   = 'observed_streamflow_data'
	station_id_name    = 'codigo'
	# id_HS_stations     = '1a02d68216f24a7fbde3669b7760652d'
	id_HS_stations     = '41787ed93210444988f807ddfaae2eea'
	func_url_build     = lambda station : url_hs +\
						'{0}/data/contents/Discharge_data/{1}.csv'.format(id_HS_stations, station)
	dict_names = {'Datetime column name'    : 'Datetime',
				  'Datetime column format'  : '%Y-%m-%d',
				  'Data column name'        : 'Streamflow (m3/s)',
				  'Data column name prefix' : 's_'}

	Update_historical_observed_data(pgres_table_name, 
									station_table_name,
									station_id_name,
									url_hs,
									id_HS_stations,
									func_url_build,
									dict_names)
	# """


	print(' Waterlevel '.center(70, '-'))
	# For waterlevel download
	station_table_name = 'stations_waterlevel'
	pgres_table_name   = 'observed_waterlevel_data'
	station_id_name    = 'codigo'
	id_HS_stations     = '41787ed93210444988f807ddfaae2eea'
	func_url_build     = lambda station : url_hs +\
						'{0}/data/contents/Waterlevel_data/{1}.csv'.format(id_HS_stations, station)
	dict_names = {'Datetime column name'    : 'Datetime',
				  'Datetime column format'  : '%Y-%m-%d',
				  'Data column name'        : 'Water Level (cm)',
				  'Data column name prefix' : 's_'}

	Update_historical_observed_data(pgres_table_name, 
									station_table_name,
									station_id_name,
									url_hs,
									id_HS_stations,
									func_url_build,
									dict_names)


	print(' Data database updated. '.center(70, '-'))
