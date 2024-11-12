import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

fld = "z500"
idxes = {"u10":0, "z500":14, "2m_temperature":2, "v10":1, "t850":5, "tp":0}
c = idxes[fld]
config1 = 'afno_backbone_finetune/saveckpt_0'
config2 = 'afno_backbone_finetune/saveckpt_1'
config3 = 'afno_backbone_finetune/saveckpt_2'
config4 = 'afno_backbone_finetune/saveckpt_3'
config5 = 'afno_backbone_finetune/saveckpt_4'

basepath = '/pscratch/sd/s/shas1693/results/era5_wind'
filenames = [config1 + "/autoregressive_predictions_"+fld+".h5"]
filenames += [config2+"/autoregressive_predictions_"+fld+".h5"]
filenames += [config3+"/autoregressive_predictions_"+fld+".h5"]
filenames += [config4+"/autoregressive_predictions_"+fld+".h5"]
filenames += [config5+"/autoregressive_predictions_"+fld+".h5"]

if fld == "tp":
    scale = 1E3 # convert rmse to mm
else:
    scale = 1

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
lns = ["o-","o-","o-","o-","o-"] # line colors for plots apart from ifs
colors = ["r","g","m","k","y"]
nms = ["","","","",""] # "_coarse" to use the coarse acc vals from inference.py

labels = ["0","1","2","3","4"]

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
    ifs = os.path.join(basepath, "ifs_2018_"+fld+"_skip0.h5")
    if plot_weyn:
        ifs = os.path.join(basepath, "ifs_2018_"+fld+"_skip0_coarse_dc.h5")
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

    with h5py.File(path_to_h5, "r") as f1:
        acc = f1["acc"+nms[idx]][:]
        rmse = f1["rmse"+nms[idx]][:]
    nic = acc.shape[0]
    
    remove_ics = []
    for ic in range(nic):
        ric = rmse[ic]
        mn = np.mean(ric[start:end,c], axis=0)
        if mn > 800:
            remove_ics.append(ic)

    print("removing {}".format(remove_ics))
    numdel = 0
    for ic in remove_ics:
        acc=np.delete(acc, ic-numdel, axis=0)
        rmse=np.delete(rmse, ic-numdel, axis=0)
        numdel += 1
    print(acc.shape)

    acc_mean = np.mean(acc[:,start:end,c], axis=0) # mean over all ics
    rmse_mean = np.mean(rmse[:,start:end,c], axis=0)*scale # mean over all ics
    acc_std = np.std(acc[:,start:end,c], axis=0) # std over all ics
    rmse_std = np.std(rmse[:,start:end,c], axis=0)*scale # std over all ics
    acc_q1 = np.quantile(acc[:,start:end,c], 0.25, axis=0) # 1st quantile
    acc_q3 = np.quantile(acc[:,start:end,c], 0.75, axis=0) # 3rd quantile
    rmse_q1 = np.quantile(rmse[:,start:end,c], 0.25, axis=0)*scale # 1st quantile
    rmse_q3 = np.quantile(rmse[:,start:end,c], 0.75, axis=0)*scale # 3rd quantile
    hrs = np.arange(6, acc_mean.shape[0]*6+6, 6)


#    for i,h in enumerate(hrs):
#        print("{},{}".format(h, rmse_mean[i]))

    ax[0].errorbar(hrs, acc_mean, fmt=lns[idx], label=labels[idx], ms=4, lw=0.7, color=colors[idx])
    ax[0].fill_between(hrs, acc_q1, acc_q3, alpha=0.25, color=colors[idx])
    ax[1].errorbar(hrs, rmse_mean, fmt=lns[idx], label=labels[idx], ms=4, lw=0.7, color=colors[idx])
    ax[1].fill_between(hrs, rmse_q1, rmse_q3, alpha=0.25, color=colors[idx])
    
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
file_nm = os.path.join(*[basepath, "ckpt_fourcastnet_accrmse_"+fld+".pdf"])
print("saving ", file_nm)
fig.savefig(file_nm, format="pdf", dpi=1200, bbox_inches="tight")
#fig.savefig(file_nm.replace(".pdf",".svg"), format="svg", dpi=1200, bbox_inches="tight")

