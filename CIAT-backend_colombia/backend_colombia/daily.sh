#! /bin/bash

# Daily bash file

cd ~/colombia-tethys-apps/CIAT-backend_colombia/backend_colombia
eval "$(conda shell.bash hook)"
conda activate gess

python r_forecast_record_db.py
python r_forecast_db.py && python r_analysis_forecast.py && python r_analysis_raw_forecast.py

