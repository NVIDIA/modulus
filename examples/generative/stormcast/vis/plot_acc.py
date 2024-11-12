import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
from ruamel.yaml import YAML


# get the 73var channel indices
with open(os.path.abspath('./config/cos_zenith_sfnonet.yaml')) as _file:
    idxs_73var = YAML().load(_file)['sfno_73ch_cos_zenith']['channel_names']

fldstr = "z500" # since filenames have z500 in them
fld = "u10m"
idxes = {"u10":0, "z500":14, "2m_temperature":2, "v10":1, "t850":5, "tp":0, "u10m":0, "t2m":2}
config1 = 'swin_73var_ft/year2018'
c = idxes[fld]
fldifs = fld if fld != 't2m' else "2m_temperature"


# add other configs to plot if needed
#config2 = 'pretrained_two_step_afno_20ch_bs_64_lr1em4_blk_8_patch_8_cosine_sched/2'
config2 = 'sfno_73ch/year2018' #'swin_73var_p4_wr80_e768_d24_dpr01_lr1em3_noz_pos_roll/year2018'
#config2 = 'afno_backbone_26var_lamb_embed1536_dpr03_twoyears_dt4/0'
config3 = 'swin_73var_zen_abspos_ft/year2018'

#/pscratch/sd/s/shas1693/results/sfno/sfno_73ch/year2018/autoregressive_predictions_z500.h5
#/pscratch/sd/s/shas1693/results/era5_wind/swin_73var_p4_wr40_lr1em3/year2018/autoregressive_predictions_z500.h5
basepath = '/pscratch/sd/s/shas1693/results/weather/era5_swin'
bsfno = '/pscratch/sd/s/shas1693/results/sfno'
#basepath = '/global/cfs/cdirs/dasrepo/shashank/fcn/'
filenames = [config1 + "/autoregressive_predictions_"+fldstr+".h5"]
filenames += [config2+"/autoregressive_predictions_"+fldstr+".h5"]
filenames += [config3+"/autoregressive_predictions_"+fldstr+".h5"]


if fld == "tp":
    scale = 1E3 # convert rmse to mm
else:
    scale = 1

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
lns = ["o-", "o-", "o-"] # line colors for plots apart from ifs
colors = ["r", "g", "m"]
nms = ["", "", ""] # "_coarse" to use the coarse acc vals from inference.py

#labels = ["FourCastNet p=8", "26var_lamb_e1536_dpr03dt4_extrayrs", "26var_lamb_dt4_lr1em3_twoyears"]
labels = ["swin", "SFNO 11-step", "swin abspos coszen"]

# weyn;s data
plot_weyn = False # plot the results from weyn's paper
plot_ifs = True
if plot_weyn:
    weyn = "/pscratch/sd/p/pharring/weyn_data"
    acc_w = np.genfromtxt(os.path.join(weyn, "acc_" + fld + ".csv"), delimiter=", ")
    acc_w = acc_w[acc_w[:,0] <= 9]
    hrs_acc_w = acc_w[:,0]*24
    acc_w = acc_w[:,1]
    rmse_w = np.genfromtxt(os.path.join(weyn, "rmse_" + fld + ".csv"), delimiter=", ")
    rmse_w = rmse_w[rmse_w[:,0] <= 9]
    hrs_rmse_w = rmse_w[:,0]*24
    rmse_w = rmse_w[:,1]
    ax[0].plot(hrs_acc_w, acc_w, "ok--", label="Weyn " + fld)
    ax[1].plot(hrs_rmse_w, rmse_w, "ok--", label="Weyn " + fld)

start = 1
end = 34 if fld != "tp" else 14
if plot_ifs:
    fldifs = fldifs.replace('m', '') if fldifs == 'u10m' else fldifs
    ifs = os.path.join('/pscratch/sd/s/shas1693/results/era5_wind/', "ifs_2018_"+fldifs+"_skip0.h5")
    with h5py.File(ifs, "r") as f:
        ifs_acc = f["acc"][:]
        ifs_rmse = f["rmse"][:]
        if fld == "tp":
            ifs_tqe = f["tqe"][:]
        nic = ifs_acc.shape[0]
        ifs_acc_mean = np.mean(ifs_acc[:,start:end,0], axis=0) # mean over all ics
        ifs_rmse_mean = np.mean(ifs_rmse[:,start:end,0], axis=0)*scale # mean over all ics
        ifs_acc_std = np.std(ifs_acc[:,start:end,0], axis=0) # std over all ics
        ifs_acc_q1 = np.quantile(ifs_acc[:,start:end,0], 0.25, axis=0) # 1st quantile
        ifs_acc_q3 = np.quantile(ifs_acc[:,start:end,0], 0.75, axis=0) # 3rd quantile


        ifs_rmse_std = np.std(ifs_rmse[:,start:end,0], axis=0)*scale # std over all ics
        ifs_rmse_q1 = np.quantile(ifs_rmse[:,start:end,0], 0.25, axis=0)*scale # 1st quantile
        ifs_rmse_q3 = np.quantile(ifs_rmse[:,start:end,0], 0.75, axis=0)*scale # 3rd quantile
        if fld == "tp":
            ifs_tqe_mean = np.mean(ifs_tqe[:,start:end,0], axis=0) # mean over all ics
            ifs_tqe_std = np.std(ifs_tqe[:,start:end,0], axis=0) # mean over all ics

    hrs = np.arange(6, ifs_acc_mean.shape[0]*6+6, 6)
    
    ax[0].errorbar(hrs, ifs_acc_mean, fmt="o-", label="IFS "+fld, ms=4, lw=0.7, color='b')
    ax[0].fill_between(hrs, ifs_acc_q1, ifs_acc_q3, alpha=0.25)
    ax[1].errorbar(hrs, ifs_rmse_mean, fmt="o-", label="IFS "+fld, ms=4, lw=0.7, color='b')
    ax[1].fill_between(hrs, ifs_rmse_q1, ifs_rmse_q3, alpha=0.25)

for idx, f in enumerate(filenames):
    path_to_h5 = os.path.join(*[basepath, f])
    if idx == 1:
        path_to_h5 = os.path.join(*[bsfno, f])
    c = idxs_73var.index(fld) if idx >= 0 else idxes[fld]

    with h5py.File(path_to_h5, "r") as f:
        acc = f["acc"+nms[idx]][:]
        rmse = f["rmse"+nms[idx]][:]
    nic = acc.shape[0]
    acc_mean = np.mean(acc[:,start:end,c], axis=0) # mean over all ics
    rmse_mean = np.mean(rmse[:,start:end,c], axis=0)*scale # mean over all ics
    acc_std = np.std(acc[:,start:end,c], axis=0) # std over all ics
    rmse_std = np.std(rmse[:,start:end,c], axis=0)*scale # std over all ics
    acc_q1 = np.quantile(acc[:,start:end,c], 0.25, axis=0) # 1st quantile
    acc_q3 = np.quantile(acc[:,start:end,c], 0.75, axis=0) # 3rd quantile
    rmse_q1 = np.quantile(rmse[:,start:end,c], 0.25, axis=0)*scale # 1st quantile
    rmse_q3 = np.quantile(rmse[:,start:end,c], 0.75, axis=0)*scale # 3rd quantile
    hrs = np.arange(6, acc_mean.shape[0]*6+6, 6)
    hrs_2 = np.arange(24, acc_mean.shape[0]*24+24, 24)
    #if idx==2:
    #if idx==1 or idx==2:
    #    hrs = hrs_2

    ax[0].errorbar(hrs, acc_mean, fmt=lns[idx], label=labels[idx], ms=4, lw=0.7, color=colors[idx])
    ax[0].fill_between(hrs, acc_q1, acc_q3, alpha=0.25, color=colors[idx])
    ax[1].errorbar(hrs, rmse_mean, fmt=lns[idx], label=labels[idx], ms=4, lw=0.7, color=colors[idx])
    ax[1].fill_between(hrs, rmse_q1, rmse_q3, alpha=0.25, color=colors[idx])
    
#endh = 2160 if fld != "tp" else 90
#xlist = np.arange(0,endh,864)
endh = 200 if fld != "tp" else 90
xlist = np.arange(0,endh,24)
fsz = "15"
ax[0].legend()
ax[1].legend()
ax[0].set_xlim(0, hrs[-1])
ax[1].set_xlim(0, hrs[-1])
ax[0].set_xlabel("forecast time (in hrs)", fontsize=fsz)
ax[1].set_xlabel("forecast time (in hrs)", fontsize=fsz)
ax[0].set_ylabel("ACC", fontsize=fsz)
ax[1].set_ylabel("RMSE", fontsize=fsz)
ax[0].set_xticks(xlist)
ax[1].set_xticks(xlist)
ax[0].tick_params(axis='both', which='both', labelsize=12)
ax[1].tick_params(axis='both', which='both', labelsize=12)
fig.tight_layout()
file_nm = os.path.join(*["./pdfs/backbone_fourcastnet_accrmse_"+fld+"_26var_24h_newstats.pdf"])
#file_nm = os.path.join(*[basepath, "backbone_fourcastnet_accrmse_"+fld+"_26var.pdf"])
print("saving ", file_nm)
plt.show()
#fig.savefig(file_nm, format="pdf", dpi=1200, bbox_inches="tight")
#fig.savefig(file_nm.replace(".pdf",".svg"), format="svg", dpi=1200, bbox_inches="tight")

