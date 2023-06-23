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

		pgres_tablename    = 'drainage'

		# ------------------- MAIN -------------------
		rv = pd.read_csv('./data/drainage_db.csv')
		rv.drop('fid', axis=1, inplace=True)

		# rv = rv[columns_to_extract].copy()
		rv.rename(columns = {col : col.lower() for col in rv.columns}, inplace = True)

		# Insert alert column
		rv['alert'] = ['R0'] * len(rv)

		# Establish connection
		db   = create_engine("postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER,
																					   pgres_password, 
																					   pgres_databasename))

		conn = db.connect()
		try:	
			# Insert to database
			rv.to_sql(pgres_tablename, con=conn, if_exists='replace', index=False)
		finally:
			# Close connection
			conn.close()


if __name__ == '__main__':
	print(' Streamflow database updating '.center(70, '-'))
	Update_drainage_db()
	print(' Streamflow database updated '.center(70, '-'))
