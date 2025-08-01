import sys

sys.path.append("../common")

import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.dataset import process_data
from common.utils import get_prior_maps

from .models import (
    HumanAttnTransformer,
)


def build(hparams, dataset_root, device, is_eval=False):
    with open(
        join(
            dataset_root,
            hparams.Data.label_file,
        ),
        "r",
    ) as json_file:
        human_scanpaths = json.load(json_file)

    human_scanpaths_all = human_scanpaths
    human_scanpaths_tp = list(
        filter(lambda x: x["condition"] == "present", human_scanpaths_all)
    )

    dataset = process_data(
        human_scanpaths,
        dataset_root,
        hparams,
        human_scanpaths_all,
    )

    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers

    train_HG_loader = DataLoader(
        dataset["gaze_train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )
    print("num of training batches =", len(train_HG_loader))

    train_img_loader = DataLoader(
        dataset["img_train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )
    valid_img_loader_TP = DataLoader(
        dataset["img_valid_TP"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
    )
    valid_HG_loader = DataLoader(
        dataset["gaze_valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
    )
    valid_HG_loader_TP = DataLoader(
        dataset["gaze_valid_TP"],
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
    )

    # Create model
    emb_size = hparams.Model.embedding_dim
    n_heads = hparams.Model.n_heads
    hidden_size = hparams.Model.hidden_dim
    if hparams.Train.use_sinkhorn:
        assert hparams.Model.separate_fix_arch, (
            "sinkhorn requires the model to be separate!"
        )

    model = HumanAttnTransformer(
        hparams.Data,
        num_decoder_layers=hparams.Model.n_dec_layers,
        hidden_dim=emb_size,
        nhead=n_heads,
        catIds=dataset["catIds"],
        num_output_layers=hparams.Model.num_output_layers,
        train_encoder=hparams.Train.train_backbone,
        train_pixel_decoder=hparams.Train.train_pixel_decoder,
        dropout=hparams.Train.dropout,
        dim_feedforward=hidden_size,
        parallel_arch=hparams.Model.parallel_arch,
        dorsal_source=hparams.Model.dorsal_source,
        num_encoder_layers=hparams.Model.n_enc_layers,
        project_queries=hparams.Train.project_queries,
        output_feature_map_name=hparams.Model.output_feature_map_name,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.Train.adam_lr, betas=hparams.Train.adam_betas
    )

    # Load weights from checkpoint when available
    if len(hparams.Model.checkpoint) > 0:
        print(
            f"loading weights from {hparams.Model.checkpoint} in {hparams.Train.transfer_learn} setting."
        )
        ckp = torch.load(join(hparams.Train.log_dir, hparams.Model.checkpoint))
        model.load_state_dict(ckp["model"])
        optimizer.load_state_dict(ckp["optimizer"])
        global_step = ckp["step"]
    else:
        global_step = 0
    if hparams.Train.parallel:
        model = torch.nn.DataParallel(model)

    human_cdf = dataset["human_cdf"]

    prior_maps_tp = get_prior_maps(
        human_scanpaths_tp, hparams.Data.im_w, hparams.Data.im_h
    )
    keys = list(prior_maps_tp.keys())
    for k in keys:
        prior_maps_tp[k] = torch.tensor(prior_maps_tp.pop(k)).to(device)

    sps_test_tp = list(filter(lambda x: x["split"] == "test", human_scanpaths_tp))

    is_lasts = [x[5] for x in dataset["gaze_train"].fix_labels]
    term_pos_weight = len(is_lasts) / np.sum(is_lasts) - 1
    print("termination pos weight: {:.3f}".format(term_pos_weight))

    return (
        model,
        optimizer,
        train_HG_loader,
        valid_HG_loader,
        train_img_loader,
        valid_img_loader_TP,
        global_step,
        human_cdf,
        prior_maps_tp,
        valid_HG_loader_TP,
        sps_test_tp,
        term_pos_weight,
        dataset["catIds"],
    )
