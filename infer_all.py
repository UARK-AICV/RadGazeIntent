import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from common.config import JsonConfig
from common.utils import (
    transform_fixations,
)
from radgazeintent.builder import build


def evaluate(
    model,
    device,
    gazeloader,
    pa,
    log_dir=None,
    args=None,
):
    print("Eval on {} batches of fixations".format(len(gazeloader)))
    model.eval()
    all_logits = []
    all_names = []
    for loader in gazeloader:
        for batch in tqdm(loader, desc="computing eval scores"):
            img_names = batch["img_name"]

            img = batch["true_state"].to(device)
            inp_seq, inp_seq_high = transform_fixations(
                batch["normalized_fixations"],
                batch["is_padding"],
                pa,
                False,
                return_highres=True,
            )
            inp_seq = inp_seq.to(device)
            inp_padding_mask = inp_seq == pa.pad_idx
            logits = (
                model(
                    img,
                    inp_seq,
                    inp_padding_mask,
                    inp_seq_high.to(device),
                )["outputs_logit"]
                .detach()
                .cpu()
            )
            all_logits.append(logits)
            all_names.append(img_names)

    all_logits = torch.cat(all_logits, dim=0)
    B, L, D = all_logits.shape
    all_logits = torch.sigmoid(all_logits) >= 0.5
    all_logits = all_logits.detach().cpu().numpy().astype(np.int64)
    all_names = np.concatenate(all_names).tolist()
    catIds = gazeloader[0].dataset.catIds
    idx_to_name = {idx: name for name, idx in catIds.items()}
    # luu lam dcm to findings de khi loop over ground truth minh co the lay tung item vao index thang vao` dict nay
    dict_res = {}
    for ni, name in enumerate(all_names):
        dict_res[name] = []
        for di in range(D):
            item = all_logits[ni, :, di].tolist()
            tmp = []
            c = 0
            for pi in item:
                if pi == 0:
                    tmp.append("null")
                else:
                    tmp.append(idx_to_name[di])
                    c = 1
            if c == 1:  # only skip when nothing got appended
                dict_res[name].append(tmp)

    import json

    with open(os.path.join(log_dir, "infer_" + pa.label_file), "w") as f:
        json.dump(dict_res, f, indent=4)
    if args is not None:
        full_json = json.load(open(os.path.join(args.dataset_root, pa.label_file)))
        # now let's make the coco formated prediction
        for item in full_json:
            if item["name"] not in dict_res:
                continue
            findings_tmp = dict_res[item["name"]]
            findings = []
            for fi in findings_tmp:
                findings.append(fi[: len(item["X"])])
            item["findings"] = findings
        with open(os.path.join(log_dir, "coco_infer_" + pa.label_file), "w") as f:
            json.dump(full_json, f, indent=4)

    return dict_res



SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparams",
        default="/home/ptthang/segmentgaze/notes/chapter_1_baseline/RadGazeIntent/configs/coco_search18_dense_SSL_TP.json",
        type=str,
        help="hyper parameters config file path",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/ptthang/segmentgaze/data_segmentgaze/all_dicom_ids_both_egd_reflacx/224",
        type=str,
        help="dataset root path",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="gpu id (default=0)")
    return parser.parse_args()


def log_dict(writer, scalars, step, prefix):
    for k, v in scalars.items():
        writer.add_scalar(prefix + "/" + k, v, step)


if __name__ == "__main__":
    args = parse_args()
    hparams = JsonConfig(args.hparams)
    dir = os.path.dirname(args.hparams)
    hparams_tp = JsonConfig(args.hparams)

    dataset_root = args.dataset_root
    if dataset_root[-1] == "/":
        dataset_root = dataset_root[:-1]
    device = torch.device(f"cuda:{args.gpu_id}")
    args.eval_only = True
    (
        model,
        optimizer,
        train_gaze_loader,
        val_gaze_loader,
        train_img_loader,
        valid_img_loader_tp,
        global_step,
        human_cdf,
        prior_maps_tp,
        valid_gaze_loader_tp,
        sps_test_tp,
        term_pos_weight,
        _,
    ) = build(hparams, dataset_root, device, args.eval_only)

    log_dir = hparams.Train.log_dir
    evaluate(
        model,
        device,
        [train_gaze_loader, val_gaze_loader],
        hparams_tp.Data,
        log_dir=log_dir,
        args=args,
    )
