import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


class Update_fews_stations():
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
        
        # Columns to remove
        rm_columns = []
        #rm_columns = ['umbralsen', 'umbralobs', 'ultimonivelsen', 'ultimonivelobs'] #, 
                      #'uroja', 'unaranja', 'uamarilla', 'ubajos', 'umaxhis']
        url_fews_station = 'http://fews.ideam.gov.co/colombia/data/ReporteTablaEstaciones.csv'
        pgres_tablename = 'stations_fews'

        # ------------------------ MAIN ------------------------
        # Import historic stations db
        fews_file_dir = os.path.join(os.path.dirname(__file__), 'data', 'station_FEWS.csv')
        his_fews_station = pd.read_csv( fews_file_dir, index_col=False)
        
        try:
            # Import stations
            cur_fews_station = pd.read_csv(url_fews_station)
            cur_fews_station.drop(columns=rm_columns, inplace=True)
            cur_fews_station.columns = [col.lower() for col in cur_fews_station.columns]

            # Build new historical data
            his_fews_station = his_fews_station.merge(right=cur_fews_station, 
                                                      how='outer', 
                                                      on='id',
                                                      suffixes = ['', '_new'])

            # Fix new historical data
            col_to_fix = [col.replace('_new', '') for col in his_fews_station.columns if '_new' in col]

            for col in col_to_fix:
                # Update the new stations in the historical data
                m = (his_fews_station[col].isnull()) | (his_fews_station[col] != his_fews_station[col + '_new'])
                his_fews_station[col].mask(m, his_fews_station[col + '_new'], inplace=True)
                his_fews_station.drop([col + '_new'], axis=1, inplace=True)

            # Fix type columns
            his_fews_station = his_fews_station.astype({col : str for col in his_fews_station.columns})
            his_fews_station = his_fews_station.astype({'lng' : float, 'lat' : float})

            his_fews_station.columns = [col.lower() for col in his_fews_station.columns]
            
            # Fix alert column
            his_fews_station['r_obs'] = his_fews_station['umbralobs'].map(self.reclass_umb)
            his_fews_station['r_sen'] = his_fews_station['umbralsen'].map(self.reclass_umb)
            his_fews_station['alert'] = his_fews_station[['r_obs', 'r_sen']].max(axis=1)
            his_fews_station['alert'] = his_fews_station['alert'].map(self.reclass_alert)

            his_fews_station.drop(['r_obs', 'r_sen'], axis=1, inplace=True)

        finally:

            # Save historical station data
            his_fews_station.to_csv(fews_file_dir, index=False)

            # Update postgress database
            # Establish connection
            db   = create_engine("postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER,
                                                                                           DB_PASS, 
                                                                                           DB_NAME))

            conn = db.connect()
            try:	
                # Insert to database
                his_fews_station.to_sql(pgres_tablename, con=conn, if_exists='replace', index=False)
            finally:
                # Close connection
                conn.close()

            db.dispose()
            

    def reclass_umb(self, x):
        if x == 'Roja':
            return 3
        elif 'Naranja' == x:
            return 2
        elif 'Amarilla' == x:
            return 1
        elif 'Nivel Bajo' == x:
            return -1
        elif 'nan':
            return np.nan
        else:
            return 9999

    def reclass_alert(self, x):
        if -1 == x:
            return '-bass'
        elif 1 == x:
            return '-Y'
        elif 2 == x:
            return '-O'
        elif 3 == x:
            return '-R'
        else:
            return 'R0'


if __name__ == "__main__":
    print(' Start upload FEWS stations '.center(70, '-'))
    Update_fews_stations()
    print(' End upload FEWS stations '.center(70, '-'))