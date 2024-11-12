import xarray as xr
import numpy as np
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import metrics
from datetime import datetime, timedelta
import machine_info
import matplotlib.pyplot as plt
import zarr
import pandas as pd

model_shortname = "diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed_v2_noema_16M"
#model_shortname = "diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed_noema"

def get_scores(score_date):

    year, month, day, hour = score_date.year, score_date.month, score_date.day, score_date.hour

    analysis_path = "/data/realtime_hrrr/zarrfiles/analysis/analysis.zarr"
    #machine_info.hrrr_forecasts_path + "hrrr_" + f"{year}{month:02d}{day:02d}_{hour:02d}z_anl.zarr"

    analysis = xr.open_zarr(analysis_path)

    analysis = analysis.sel(time=slice(score_date + timedelta(hours=1), score_date + timedelta(hours=11)))

    ml_forecast_path = machine_info.ml_forecasts_path + "/" + model_shortname + "/" + f"ml_{year}{month:02d}{day:02d}_{hour:02d}z.zarr"

    hrrr_forecast_path = machine_info.hrrr_forecasts_path + "hrrr_" + f"{year}{month:02d}{day:02d}_{hour:02d}z_forecast.zarr"

    #ml_forecast = zarr.open(ml_forecast_path, mode='r')['edm_prediction']['refc'][0:12]

    hrrr_forecast = xr.open_zarr(hrrr_forecast_path).sel(time = analysis.time.values)

    ml_forecast = xr.open_zarr(ml_forecast_path)['data'].sel(time=analysis.time.values).values[:, 0, :, :]

    #hrrr_forecast = hrrr_forecast.sel(time=slice(score_date, score_date + timedelta(hours=12)))

    analysis = analysis.load()

    hrrr_forecast = hrrr_forecast.load()

    fss = metrics.fraction_skill_score(analysis, hrrr_forecast)

    ml_forecast_ds = xr.Dataset(
                                data_vars=dict(REFC=(['time', 'y', 'x'], ml_forecast.astype(np.float32))), 
                                coords=dict(time=hrrr_forecast.time.values,
                                            longitude=(['y', 'x'], hrrr_forecast.longitude.values),
                                            latitude=(['y', 'x'], hrrr_forecast.latitude.values)))

    fss_ml = metrics.fraction_skill_score(analysis, ml_forecast_ds)

    return fss, fss_ml


def plot_scores(fss, fss_ml, datestr=""): 

    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green']

    for idx, cutoff in enumerate(fss.cutoff.values):

        fss_cutoff = fss.sel(cutoff=cutoff)
        time = fss.time.values
        ax.plot(time, fss['REFC'].values[idx], label=f"fss cutoff={cutoff} hrrr", color=colors[idx], marker='o')
        ax.plot(time, fss_ml['REFC'].values[idx], label=f"fss cutoff={cutoff} diffusion", linestyle='dashed', color=colors[idx], marker='o')
    
    #ticks at 45 degrees
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    #expand the figure size to fit the x labels
    fig.autofmt_xdate()

    ax.legend()
    ax.set_ylabel("FSS")
    ax.set_xlabel("Time")
    ax.set_title("Fractional Skill Score comparison \n {} \n date range: {}".format(model_shortname, datestr))
    fig.tight_layout()
    plt.savefig(f"fss_plots/fss_{datestr}_{model_shortname}.png")

def temporal_mean(begin_date, end_date):

    datetime_list = []
    n_forecasts = 0

    datestr = f"{begin_date.year}{begin_date.month:02d}{begin_date.day:02d}_{begin_date.hour:02d}z-{end_date.year}{end_date.month:02d}{end_date.day:02d}_{end_date.hour:02d}z"

    while begin_date < end_date:
            
            datetime_list.append(begin_date)
            begin_date += timedelta(hours=6)

    for initialization in datetime_list:

        try:
            if "mean_fss_hrrr" not in locals():
                fss_hrrr, fss_ml = get_scores(initialization)
                mean_fss_hrrr = fss_hrrr['REFC'].values
                mean_fss_ml = fss_ml['REFC'].values
                cutoffs = fss_hrrr.cutoff.values
                time_ = np.arange(1, len(fss_hrrr.time.values) + 1)
            else:
                fss, fss_ml = get_scores(initialization)
                mean_fss_hrrr += fss['REFC'].values
                mean_fss_ml += fss_ml['REFC'].values
            n_forecasts += 1
        except:
            print(f"Error at {initialization}")
            continue

    
    mean_fss_hrrr /= n_forecasts
    mean_fss_ml /= n_forecasts


    mean_fss_hrrr_ds = xr.Dataset(
                                data_vars=dict(REFC=(['cutoff', 'time'], mean_fss_hrrr.astype(np.float32))),
                                coords=dict(cutoff=cutoffs,
                                            time=time_))
                                        
    mean_fss_ml_ds = xr.Dataset(
                                data_vars=dict(REFC=(['cutoff', 'time'], mean_fss_ml.astype(np.float32))),
                                coords=dict(cutoff=cutoffs,
                                            time=time_))
                                    
    plot_scores(mean_fss_hrrr_ds, mean_fss_ml_ds, datestr)





if __name__ == '__main__':

    #initial_date = datetime(2024, 5, 3, 0)
    #final_date = datetime(2024, 5, 17, 0)

    #while initial_date < final_date:

    #    try:
    #        hrrr, ml = get_scores(initial_date)

    #        datestr = initial_date.strftime("%Y%m%d_%H")
    #        plot_scores(hrrr, ml, datestr)
    #    except:
    #        print(f"Error at {initial_date}")

    #    initial_date += timedelta(hours=6)

    #score_date = datetime(2024, 5, 14, 12)
    #hrrr, ml = get_scores(score_date)

    #datestr = score_date.strftime("%Y%m%d_%H")

    #plot_scores(hrrr, ml, datestr)

    temporal_mean(datetime(2024, 5, 3, 0), datetime(2024, 5, 17, 0))