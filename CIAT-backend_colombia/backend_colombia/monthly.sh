#! /bin/bash

# monthly bash file 

cd ~/CIAT-backend_colombia/backend_colombia
eval "$(conda shell.bash hook)"
conda activate gess

python r_historical_simulation_db.py

