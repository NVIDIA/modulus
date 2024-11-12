#!/bin/bash

docker run \
  --gpus all \
  -v /home/jpathak/Data:/data \
  -v /home/jpathak/Scripts/fcn-dev-hrrr:/code \
  --workdir /code/ \
  localdev:latest \
  bash -c "pip install metpy; \
  ldconfig; \
  python realtime/obtain_datasets.py"

#push forecast to gitlab dashboard
bash -c "cp /home/jpathak/Scripts/fcn-dev-hrrr/current_run/current_forecast.gif /home/jpathak/Scripts/hrrr-dashboard/forecast/; \
  cp /home/jpathak/Scripts/fcn-dev-hrrr/current_run/current_forecast_ensemble.gif /home/jpathak/Scripts/hrrr-dashboard/forecast/; \
  cd /home/jpathak/Scripts/hrrr-dashboard/; \
  git add -A; \
  git commit -m 'Update forecast'; \
  git push dashboard;"
#  git push ext_dashboard" 

#sync hrrr and gfs data to pdx
source ~/.swiftstack
AWS_ACCESS_KEY_ID=team-earth2-datasets AWS_SECRET_ACCESS_KEY=$PDXSECRET S3_ENDPOINT_URL=https://pdx.s8k.io /home/jpathak/datatransfer/s5cmd sync /home/jpathak/Data/realtime_hrrr s3://us_km_ddwp/
AWS_ACCESS_KEY_ID=team-earth2-datasets AWS_SECRET_ACCESS_KEY=$PBSSSECRET S3_ENDPOINT_URL=https://pbss.s8k.io /home/jpathak/datatransfer/s5cmd sync /home/jpathak/Data/realtime_hrrr s3://us_km_ddwp/


