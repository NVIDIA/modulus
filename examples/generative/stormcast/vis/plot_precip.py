''' plot precip inference metrics '''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import h5py
import os

def tp_log(img, eps=1E-5):
    img = np.log1p(img/eps)
    return img

def tp_unlog(x, eps=1E-5):
    return eps*(np.exp(x)-1)

def compute_acc(gt, pred, ifs):
    gt = tp_unlog(gt)
    pred = tp_unlog(pred)
    ifs = tp_unlog(ifs)


def plot_rmse(rmse, fig, ax, path):
    rmse_mean = np.mean(rmse[:,0:,0], axis=0)*1E3 # mean over all ics in mm
    hrs = np.arange(0, rmse_mean.shape[0]*6, 6)
    ax.plot(hrs, rmse_mean, "r-")
    ax.set_xlabel("time (in hrs)")
    ax.set_ylabel("rmse (in mm)")
    fig.tight_layout()
    fig.savefig(path, format="pdf", dpi=1200, bbox_inches="tight")

def plot_acc(acc, fig, ax, path):
    acc_mean = np.mean(acc[:,0:,0], axis=0) # mean over all ics
    hrs = np.arange(0, acc_mean.shape[0]*6, 6)
    ax.plot(hrs, acc_mean, "r-")
    ax.set_xlabel("time (in hrs)")
    ax.set_ylabel("acc")
    fig.tight_layout()
    fig.savefig(path, format="pdf", dpi=1200, bbox_inches="tight")

def plot_tqe(tqe, fig, ax, path):
    tqe_mean = np.mean(tqe[:,0:,0], axis=0) # mean over all ics
    hrs = np.arange(0, tqe_mean.shape[0]*6, 6)
    ax.plot(hrs, tqe_mean, "r-")
    ax.set_xlabel("time (in hrs)")
    ax.set_ylabel("tqe (in mm)")
    fig.tight_layout()
    fig.savefig(path, format="pdf", dpi=1200, bbox_inches="tight")

def plot_precip(gt, pred, ifs, ax):
    scale = np.abs(gt).max()
    ax[0].imshow(gt, cmap="coolwarm", norm=Normalize(0.,scale))
    ax[0].set_title("tar")
    ax[0].axis('off')
    ax[1].imshow(pred, cmap="coolwarm", norm=Normalize(0.,scale))
    ax[1].set_title("pred")
    ax[1].axis('off')
    ax[2].imshow(ifs, cmap="coolwarm", norm=Normalize(0.,scale))
    ax[2].set_title("ifs")
    ax[2].axis('off')

def plot_pdf(gt, pred, ifs, ax):
    tar_hist, bin_edges = np.histogram(gt, bins=40)
    gen_hist, _ = np.histogram(pred, bins=bin_edges)
    ifs_hist, _ = np.histogram(ifs, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax[3].errorbar(centers, tar_hist, fmt='ks--', label='tar', markersize=2)
    ax[3].errorbar(centers, gen_hist, fmt='ro-', label='pred', markersize=2)
    ax[3].errorbar(centers, ifs_hist, fmt='bo-', label='ifs', markersize=2)
    ax[3].set_yscale('log')
    ax[3].legend()
    ax[3].set_xlabel('bins')
    ax[3].set_ylabel('frequency')
#    ax[2].axis('off')

def plot_quantile(gt, pred, ifs, ax):
    qs = 100
    qlim = 5
    qcut = 0.1
    qtile = 1. - np.logspace(-qlim, -qcut, num=qs)
    P_gt = np.quantile(gt.flatten(), qtile)
    P_pred = np.quantile(pred.flatten(), qtile)
    P_ifs = np.quantile(ifs.flatten(), qtile)
    qtiles = np.logspace(-qlim, -qcut, num=qs)
    ax[4].plot(qtiles, P_gt, 'k--', label='tar')
    ax[4].plot(qtiles, P_pred, 'r-', label='pred')
    ax[4].plot(qtiles, P_ifs, 'b-', label='ifs')
    ax[4].set_xlim((qtiles[-1], qtiles[0]))
    ax[4].set_xscale('log')
    ax[4].set_xticks(ticks=10.**np.arange(-1,-qlim - 1, -1))
    ax[4].set_xticklabels(labels=['%g%%'%(100.*(1. - 10.**q)) for q in np.arange(-1,-qlim - 1, -1)])
    ax[4].set_xlabel('percentile')
    ax[4].set_ylabel('total 6h precipitation (mm)')
    ax[4].legend()

def vis_precip(gt, pred, ifs, fig, ax, path):
    gt = tp_log(gt)
    pred = tp_log(pred)
    ic = 0 # which ic
    assert ic%2 == 0 #ifs data has 12hr increments for ics
    ic_t1 = ic//2
    print(ifs.shape)
    ifs = ifs[ic_t1]
    ifs = tp_log(ifs)
    times = [1, 4, 8]
    # 1 day = 24*1/6
    for idx in range(3):
        plot_precip(gt[ic,times[idx],0], pred[ic,times[idx],0], ifs[times[idx]], ax[idx])
        plot_pdf(gt[ic,times[idx],0], pred[ic,times[idx],0], ifs[times[idx]], ax[idx])
        plot_quantile(gt[ic,times[idx],0], pred[ic,times[idx],0], ifs[times[idx]], ax[idx])

    fig.tight_layout()
    fig.savefig(path, format="pdf", dpi=1200, bbox_inches="tight")
     

config='precip'
run_num = '0'
basepath = '/pscratch/sd/s/shas1693/results/era5_wind'
filename = "autoregressive_predictions_tp.h5"

path_ifs = '/pscratch/sd/p/pharring/ERA5/precip/tigge/total_precipitation/2018.h5'
ifs = h5py.File(path_ifs, 'r')['tp']

fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20,10))
fig_acc, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(9,5))
fig_tqe, ax_tqe = plt.subplots(nrows=1, ncols=1, figsize=(9,5))
fig_rmse, ax_rmse = plt.subplots(nrows=1, ncols=1, figsize=(9,5))
path_to_h5 = os.path.join(*[basepath, config, run_num, filename])

with h5py.File(path_to_h5, "r") as f:
    acc = f["acc"][:]
    rmse = f["rmse"][:]
    gt = f["ground_truth"][:]
    pred = f["predicted"][:]
    tqe = f["tqe"][:]

res_path = os.path.join(*[basepath, config, run_num, "rmse_precip.pdf"])
print("plotting rmse...")
plot_rmse(rmse, fig_rmse, ax_rmse, res_path)
res_path = os.path.join(*[basepath, config, run_num, "acc_precip.pdf"])
print("plotting acc...")
plot_acc(acc, fig_acc, ax_acc, res_path)
res_path = os.path.join(*[basepath, config, run_num, "tqe_precip.pdf"])
print("plotting tqe...")
plot_tqe(tqe, fig_tqe, ax_tqe, res_path)
res_path = os.path.join(*[basepath, config, run_num, "vis_precip.pdf"])
print("plotting precip vis...")
vis_precip(gt, pred, ifs, fig, ax, res_path)

