import os
from dotenv import load_dotenv
import sys
import psycopg2
import numpy as np
import pandas as pd
import datetime as dt
from sqlalchemy import create_engine

from backend_auxiliar import get_data_wfs, get_station_info

########################################################################
#                                                                      #
#                           r_station_db.py                            #
#                                                                      #
########################################################################

class Update_station_db:

	def __init__(self, name_file, code_name_col, reach_name_col, pgres_tablename, x_col, y_col):
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

		# Identifies
		main_identifiers = {'Station code' : 'codigo',
							'Reach code'   : 'comid'}		

		# ---------- MAIN ----------
		# Establish connection
		db   = create_engine("postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER,
																					   pgres_password, 
																					   pgres_databasename))

		# Build directory
		path_data = os.path.join(os.path.sep.join(__file__.split(os.path.sep)[:-1]), 'data')

		# Read main data
		rv = pd.read_csv(os.path.join(path_data, name_file))
		rv = self.__fix_dataframe__(rv, x_col=x_col, y_col=y_col)

		# Load extra info in station
		station_info = get_station_info()
		station_info['CODIGO'] = station_info['CODIGO'].astype(str)
		rv[code_name_col] = rv[code_name_col].astype(str)

		# Merge extra data
		rv = rv.merge(station_info, how='left', left_on=code_name_col, right_on='CODIGO')

		# Remove capitalize letters for fix columns names for postgres database
		rv.rename(columns={str(col) : str(col).lower() for col in rv.columns}, inplace=True)
		rv.dropna(subset = [code_name_col.lower()], inplace=True)

		# Fix principal identifies columns
		## Stations
		if main_identifiers['Station code'] != code_name_col.lower():
			try:
				rv.drop(main_identifiers['Station code'], axis=1, inplace = True)
				rv.rename(columns = {code_name_col.lower() : main_identifiers['Station code']}, inplace=True)
			except:
				rv.rename(columns = {code_name_col.lower() : main_identifiers['Station code']}, inplace=True)

		## Reachs
		if main_identifiers['Reach code'] != reach_name_col.lower():
			try:
				rv.drop(main_identifiers['Reach code'], axis=1, inplace = True)
				rv.rename(columns = {reach_name_col.lower() : main_identifiers['Reach code']}, inplace=True)
			except:
				rv.rename(columns = {reach_name_col.lower() : main_identifiers['Reach code']}, inplace=True)

		# Build database
		conn = db.connect()
		try:
			# Insert to database
			rv.to_sql(pgres_tablename, con=conn, if_exists='replace', index=False)
		finally:
			# Close connection
			conn.close()


	def __fix_dataframe__(self, df, x_col, y_col):
		"""
		Fix the dataframe for the IDEAM_Stations_v2
		"""
		# Build x and y column from the_geom column of wfs
		df['x'] = df[x_col]
		df['y'] = df[y_col]

		# Add Alarm column
		df['alert'] = ['R0'] * len(df)

		return df


if __name__ == '__main__':

	# Read by command the main data from the csv file
	name_file       = sys.argv[1]
	code_name_col   = sys.argv[2]
	reach_name_col  = sys.argv[3]
	pgres_tablename = sys.argv[4]
	x_col           = sys.argv[5]
	y_col           = sys.argv[6]

	print(' Station database updating. '.center(70, '-'))
	# Update station historical validation tool
	Update_station_db(name_file, code_name_col, reach_name_col, pgres_tablename, x_col, y_col)
	print(' Station database updated. '.center(70, '-'))

