import glob
import os

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

    LOG_FOUT = open(os.path.join(cfg.inference_options.save_path, "log_inf.txt"), "a")

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.inference_options.seed)
    torch.cuda.manual_seed_all(cfg.inference_options.seed)
    np.random.seed(cfg.inference_options.seed)

    # model setting
    device = torch.device("cuda" if cfg.general.cuda else "cpu")

    # Initialize compensator
    compensator = DGCNN().to(device)
    compensator.load_state_dict(
        torch.load(cfg.inference_options.generator_path, map_location="cpu"),
        strict=False,
    )
    for g in compensator.parameters():
        g.requires_grad = False  # to avoid computation
    log_string(LOG_FOUT, "Trained Shape Compensation model loaded")

    # Initialize predictor
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

    # Load parts data, in STL or OBJ format
    # todo: change the path to search all subfolders
    data_path = cfg.inference_options.data_path
    parts_ds = glob.glob(data_path + "/*.stl") + glob.glob(
        data_path + "/*.obj"
    )  # main processing
    log_string(LOG_FOUT, f"\n\nStart processing data .. {parts_ds}")

    for part_ in parts_ds:
        tic()
        log_string(LOG_FOUT, f"Processing Part {part_}")

        # load data as mesh
        cad = trimesh.load_mesh(part_)
        # convert mesh to points by taking vertices
        pts1 = torch.FloatTensor(np.asarray(cad.vertices))

        # randomize point indices
        rand = np.arange(len(pts1))
        np.random.shuffle(rand)
        pts1_full = pts1[rand]
        log_string(LOG_FOUT, f"Part full shape: {pts1_full.shape}")

        subsample = cfg.inference_options.num_points
        n = len(pts1_full) // subsample
        pred_res = []
        comp_res = []
        out_res = []
        for i in range(n + 1):
            log_string(LOG_FOUT, f"batching {i}")
            if i == n and len(pts1_full) % subsample == 0:
                log_string(LOG_FOUT, "no need to process")
                continue
            elif i == n:
                pts1 = pts1_full[-subsample:]
            else:
                pts1 = pts1_full[subsample * i : subsample * (i + 1)]

            edge_index = torch_geometric.nn.knn_graph(pts1, 20)
            if cfg.general.cuda:
                log_string(LOG_FOUT, "Use Cuda")
                pts1 = pts1.squeeze(0).to(device).squeeze(0)
                edge_index = edge_index.squeeze().to(device)

            # Compensate the original CAD, then predict the deviation from the compensated CAD
            com = compensator(torch_geometric.data.Data(x=pts1, edge_index=edge_index))
            out = discriminator(torch_geometric.data.Data(x=com, edge_index=edge_index))

            # Directly predict the deviation from the original CAD
            pre = discriminator(
                torch_geometric.data.Data(x=pts1, edge_index=edge_index)
            )

            if i == n:
                valid = len(pts1_full) - n * subsample
                com = com[-valid:]
                out = out[-valid:]
                pre = pre[-valid:]

            comp_res.append(com.detach().cpu().numpy())
            pred_res.append(pre.detach().cpu().numpy())
            out_res.append(out.detach().cpu().numpy())

        # Concatenate batches data to the final full part
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

        # Get input STL model name
        part_name = os.path.basename(part_)
        subfolder = save_path + part_name[:-4]  # + "_compensated/"

        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

        np.savetxt(
            os.path.join(subfolder, part_name[:-4] + "_cad.csv"),
            pts1_full.cpu().numpy(),
            fmt="%.8f",
            delimiter=",",
        )
        log_string(LOG_FOUT, "Wrote to CAD mesh")

        np.savetxt(
            os.path.join(subfolder, part_name[:-4] + "_comp.csv"),
            cad_comp_res,
            fmt="%.8f",
            delimiter=",",
        )
        log_string(
            LOG_FOUT, f"Wrote to Compensated pointcloud: {part_name[:-4]}_comp.csv"
        )

        cad.vertices = cad_comp_res
        cad.export(os.path.join(subfolder, part_name[:-4] + "_comp.stl"))
        log_string(LOG_FOUT, f"Wrote to Compensated mesh: {part_name[:-4]}_comp.stl")

        if cfg.inference_options.save_extra:
            log_string(LOG_FOUT, "Saving extra files...")
            np.savetxt(
                os.path.join(subfolder, part_name[:-4] + "_pred.csv"),
                cad_pred_res,
                fmt="%.8f",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(subfolder, part_name[:-4] + "_outp.csv"),
                cad_outp_res,
                fmt="%.8f",
                delimiter=",",
            )
        log_string(
            LOG_FOUT,
            "Wrote to deviation from the original CAD, and deviation from the compensated CAD",
        )

        toc()


if __name__ == "__main__":

    with initialize(config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["+db=mysql", "+db.user=me"])

        distributed_option = cfg.general.use_distributed
        device = torch.device("cuda" if cfg.general.cuda else "cpu")

    os.makedirs(cfg.inference_options.save_path, exist_ok=True)
    main()
