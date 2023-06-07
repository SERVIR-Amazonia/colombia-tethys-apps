#! /bin/bash

# Daily bash file

cd ~/CIAT-backend_colombia/backend_colombia
eval "$(conda shell.bash hook)"
conda activate gess

python r_forecast_db.py
python r_forecast_record_db.py
python r_analysis_forecast.py
# TODO : add analysis forecast no fixed

