import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import datetime as dt
from sqlalchemy import create_engine

from backend_auxiliar import get_data_wfs

########################################################################
#                                                                      #
#                          r_drainage_db.py                            #
#                                                                      #
########################################################################


class Update_drainage_db:
	def __init__(self):

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

		# Hydroshare wfs data to download
		id_HS  = 'HS-cff2657bc8244560b559320162bf8ce4'
		layer  = 'south_america-colombia-geoglows-drainage_line'
		url    = f"https://geoserver.hydroshare.org/geoserver/{id_HS}/ows"
		params = {'service'     : 'WFS',
				  'version'     : '1.0.0',
				  'request'     : 'GetFeature',
				  'typeName'    : f'{id_HS}%3A{layer}',
				  'maxFeatures' : '9999'}

		columns_to_extract = ['HydroID'] 

		# ------------------- MAIN -------------------
		# Establish connection
		db   = create_engine("postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER,
																					   pgres_password, 
																					   pgres_databasename))

		# Build hydroshare link
		full_url = url + '?' + '&'.join([ f'{key}={value}' for key, value in params.items() ])

		# Download drainage list
		rv = get_data_wfs(url   = full_url,
						  id_HS = id_HS,
						  layer = layer)
		rv = pd.DataFrame(rv)
		rv = rv[columns_to_extract].copy()
		rv.rename(columns = {col : col.lower() for col in rv.columns}, inplace = True)

		# Insert alert column
		rv['alert'] = ['R0'] * len(rv)

		try:
			conn = db.connect()
			# Insert to database
			rv.to_sql(pgres_tablename, con=conn, if_exists='replace', index=False)

		finally:
			# Close connection
			conn.close()


if __name__ == '__main__':
	print(' Streamflow database updating '.center(70, '-'))
	Update_drainage_db()
	print(' Streamflow database updated '.center(70, '-'))
