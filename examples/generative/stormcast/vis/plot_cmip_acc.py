import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

fld = "z500"
idxes_ifs = {"u10":0, "z500":14, "2m_temperature":2, "v10":1, "t850":5, "tp":0}
idxes = {"u10":37, "z500":32, "2m_temperature":36, "v10":38, "t850":None}
c = idxes[fld]


config1 = 'afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune/0'
config2 = 'afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune/era5res'

basepath = '/pscratch/sd/s/shas1693/results/climate'
filenames = [config1 + "/autoregressive_predictions_"+fld+".h5"]
filenames += [config2+"/autoregressive_predictions_"+fld+".h5"]


if fld == "tp":
    scale = 1E3 # convert rmse to mm
else:
    scale = 1

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
lns = ["o-", "o-", "o-"] # line colors for plots apart from ifs
colors = ["r", "g", "m"]
nms = ["", "", ""] # "_coarse" to use the coarse acc vals from inference.py

labels = ["HighResMIP", "ERA5"]
# weyn;s data
plot_ifs = True

start = 1
end = 34 if fld != "tp" else 14
if plot_ifs:
    ifs = os.path.join(basepath, "ifs_2018_"+fld+"_skip0.h5")
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
file_nm = os.path.join(*["./pdfs/cmip_accs_"+fld+".pdf"])
print("saving ", file_nm)
fig.savefig(file_nm, format="pdf", dpi=1200, bbox_inches="tight")
#fig.savefig(file_nm.replace(".pdf",".svg"), format="svg", dpi=1200, bbox_inches="tight")

