import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch_geometric
import trimesh
from hydra import compose, initialize

# from omegaconf import DictConfig, OmegaConf
from utils import log_string, tic, toc

from modulus.models.dgcnn.dgcnn_compensation import DGCNN

# from modulus.models.dgcnn.dgcnn_compensation import DGCNN_ocardo


def main():
    """
    With the trained ckpt for both the prediction engine, and the compensation engine, config in cfg.inference_options
        1. Load the parts to compensate from path:
            cfg.inference_options.data_path
            data path should contain stl / obj files to compensate for
        2. From the trained compensation model, compensate the input CAD
        3. From the trained deviation prediction model, predict the potential deviation of the compensated CAD, to compare with the original design
    second:
    """
    # Read the configs
    with initialize(config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["+db=mysql", "+db.user=me"])

    LOG_FOUT = open(
        os.path.join(cfg.inference_options.save_path, "log_inf.txt"), "a"
    )

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.inference_options.seed)
    torch.cuda.manual_seed_all(cfg.inference_options.seed)
    np.random.seed(cfg.inference_options.seed)

    # model setting
    device = torch.device("cpu")  # device('cuda' if args.cuda else 'cpu')
    model = DGCNN().to(device)

    model.load_state_dict(
        torch.load(cfg.inference_options.generator_path, map_location="cpu"),
        strict=False,
    )
    for g in model.parameters():
        g.requires_grad = False  # to avoid computation
    log_string(LOG_FOUT, "Trained Shape Compensation model loaded")

    discriminator = DGCNN().to(device)
    discriminator.load_state_dict(
        torch.load(cfg.inference_options.discriminator_path, map_location="cpu"),
        strict=False,
    )
    for p in discriminator.parameters():
        p.requires_grad = False  # to avoid computation
    log_string(LOG_FOUT, "Trained Shape Prediction model loaded")

    save_path = cfg.inference_options.save_path
    os.makedirs(save_path, exist_ok=True)

    # tic()
    data_path = cfg.inference_options.data_path
    models = glob.glob(data_path + "/*.stl") + glob.glob(
        data_path + "/*.obj"
    )  # main processing
    log_string(LOG_FOUT, f"Start processing data .. {models}")

    for m in models:
        tic()
        log_string(LOG_FOUT, f"processing Part {m}")

        # load data as mesh
        cad = trimesh.load_mesh(m)
        # convert mesh to points by taking vertices
        pts1 = torch.FloatTensor(np.asarray(cad.vertices))

        # randomize point indices
        rand = np.arange(len(pts1))
        np.random.shuffle(rand)
        # print(rand)
        pts1_full = pts1[rand]
        log_string(LOG_FOUT, f"Part full shape: {pts1_full.shape}")

        subsample = cfg.inference_options.num_points
        n = len(pts1_full) // subsample
        pred_res = []
        comp_res = []
        out_res = []
        for i in range(n + 1):
            print("batching %03d" % i)
            if i == n and len(pts1_full) % subsample == 0:
                print("no need to process")
                continue
            elif i == n:
                pts1 = pts1_full[-subsample:]
            else:
                pts1 = pts1_full[subsample * i : subsample * (i + 1)]
            if cfg.cuda:
                pts1 = pts1.squeeze(0).to(device).squeeze(0)
                edge_index = torch_geometric.nn.knn_graph(pts1, 20)
                edge_index = edge_index.squeeze().to(device)

            # Compensate the original CAD, then predict the deviation from the compensated CAD
            com = model(torch_geometric.data.Data(x=pts1, edge_index=edge_index))
            out = discriminator(torch_geometric.data.Data(x=com, edge_index=edge_index))

            # Directly predict the deviation from the original CAD
            pre = discriminator(
                torch_geometric.data.Data(x=pts1, edge_index=edge_index)
            )
            com = com.detach().cpu()  # + m
            out = out.detach().cpu()  # + m
            pre = pre.detach().cpu()  # + m

            if i == n:
                valid = len(pts1_full) - n * subsample
                com = com[-valid:]
                out = out[-valid:]
                pre = pre[-valid:]
                # print(com.shape)
                # print(pre.shape)
                # print(out.shape)
                # print(valid)

            cad_comp_pts = com
            cad_pred_pts = pre
            cad_outp_pts = out
            comp_res.append(cad_comp_pts.numpy())
            pred_res.append(cad_pred_pts.numpy())
            out_res.append(cad_outp_pts.numpy())
        cad_comp_pts = np.concatenate(comp_res, 0)
        cad_pred_pts = np.concatenate(pred_res, 0)
        cad_outp_pts = np.concatenate(out_res, 0)

        # note that all above results are shuffled one! back to the original order
        pts1_res = np.zeros((len(pts1_full), 3))
        cad_comp_res = np.zeros((len(cad_comp_pts), 3))
        cad_pred_res = np.zeros((len(cad_pred_pts), 3))
        cad_outp_res = np.zeros((len(cad_outp_pts), 3))

        for i in range(len(cad_comp_pts)):
            pts1_res[rand[i]] = pts1_full[i]
            cad_comp_res[rand[i]] = cad_comp_pts[i]
            cad_pred_res[rand[i]] = cad_pred_pts[i]
            cad_outp_res[rand[i]] = cad_outp_pts[i]
        print(pts1)
        print(pts1_res)
        print(cad_comp_pts.shape)

        # Get input STL model name
        modelname = os.path.basename(m)
        print("modelname: " + modelname)
        subfolder = save_path + modelname[:-4]  # + "_compensated/"
        print("subfolder: ", subfolder)

        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

        np.savetxt(
            os.path.join(subfolder, modelname[:-4] + "_cad.csv"),
            pts1_full.cpu().numpy(),
            fmt="%.8f",
            delimiter=",",
        )
        np.savetxt(
            os.path.join(subfolder, modelname[:-4] + "_comp.csv"),
            cad_comp_res,
            fmt="%.8f",
            delimiter=",",
        )

        cad.vertices = cad_comp_res
        cad.export(
            os.path.join(subfolder, modelname[:-4] + "_comp.stl")
        )  # = trimesh.load_mesh(args.data_path)

        if cfg.inference_options.save_extra:
            print("Saving extra files...")
            print(cad_pred_pts.shape)
            print(cad_outp_pts.shape)
            np.savetxt(
                os.path.join(subfolder, modelname[:-4] + "_pred.csv"),
                cad_pred_res,
                fmt="%.8f",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(subfolder, modelname[:-4] + "_outp.csv"),
                cad_outp_res,
                fmt="%.8f",
                delimiter=",",
            )

        toc()


if __name__ == "__main__":

    with initialize(config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["+db=mysql", "+db.user=me"])

        distributed_option = cfg.general.use_distributed
        device = torch.device("cuda" if cfg.general.cuda else "cpu")

    main()
