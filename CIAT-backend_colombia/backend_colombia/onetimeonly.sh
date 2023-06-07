#! /bin/bash

# One time only file
# This routine prepare the database of postgres
# is the first routine to run after instalation of
# postgres

cd ~/CIAT-backend_colombia/backend_colombia
eval "$(conda shell.bash hook)"
conda activate gess

python r_drainage_db.py
python r_station_db.py 'Selected_Stations_Colombia_Q_v0.csv' 'ID' 'new_comid' 'stations_streamflow' 'Longitude' 'Latitude'
python r_station_db.py 'Selected_Stations_Colombia_WL_v4.csv' 'CODIGO' 'new_COMID' 'stations_waterlevel' 'longitud' 'latitud'
python r_observed_data_db.py
