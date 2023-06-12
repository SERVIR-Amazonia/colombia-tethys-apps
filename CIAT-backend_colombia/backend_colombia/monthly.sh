#! /bin/bash

# monthly bash file 

cd /home/jrc/colombia-tethys-apps/CIAT-backend_colombia/backend_colombia
eval "$(conda shell.bash hook)"
conda activate gess

python r_historical_simulation_db.py

