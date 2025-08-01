import sys

sys.path.append("../common")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.distributions import Categorical
from tqdm import tqdm

from common import utils
from common.utils import (
    transform_fixations,
)


def get_IOR_mask(norm_x, norm_y, h, w, r):
    bs = len(norm_x)
    x, y = norm_x * w, norm_y * h
    Y, X = np.ogrid[:h, :w]
    X = X.reshape(1, 1, w)
    Y = Y.reshape(1, h, 1)
    x = x.reshape(bs, 1, 1)
    y = y.reshape(bs, 1, 1)
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    return torch.from_numpy(mask.reshape(bs, -1))


def scanpath_decode(model, img, task_ids, pa, sample_action=False, center_initial=True):
    bs = img.size(0)
    with torch.no_grad():
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = model.encode(img)
    if center_initial:
        normalized_fixs = torch.zeros(bs, 1, 2).fill_(0.5)
        action_mask = get_IOR_mask(
            np.ones(bs) * 0.5, np.ones(bs) * 0.5, pa.im_h, pa.im_w, pa.IOR_radius
        )
    else:
        normalized_fixs = torch.zeros(bs, 0, 2)
        action_mask = torch.zeros(bs, pa.im_h * pa.im_w)

    stop_flags = []
    for i in range(pa.max_traj_length):
        with torch.no_grad():
            if i == 0 and not center_initial:
                ys = ys_high = torch.zeros(bs, 1).to(torch.long)
                padding = torch.ones(bs, 1).bool().to(img.device)
            else:
                ys, ys_high = utils.transform_fixations(
                    normalized_fixs, None, pa, False, return_highres=True
                )
                padding = None

            out = model.decode_and_predict(
                dorsal_embs.clone(),
                dorsal_pos,
                dorsal_mask,
                high_res_featmaps,
                ys.to(img.device),
                padding,
                ys_high.to(img.device),
                task_ids,
            )
            prob, stop = out["pred_fixation_map"], out["pred_termination"]
            prob = prob.view(bs, -1)
            stop_flags.append(stop)

            if pa.enforce_IOR:
                # Enforcing IOR
                batch_idx, visited_locs = torch.where(action_mask == 1)
                prob[batch_idx, visited_locs] = 0

        if sample_action:
            m = Categorical(prob)
            next_word = m.sample()
        else:
            _, next_word = torch.max(prob, dim=1)

        next_word = next_word.cpu()
        norm_fy = (next_word // pa.im_w) / float(pa.im_h)
        norm_fx = (next_word % pa.im_w) / float(pa.im_w)
        normalized_fixs = torch.cat(
            [normalized_fixs, torch.stack([norm_fx, norm_fy], dim=1).unsqueeze(1)],
            dim=1,
        )

        new_mask = get_IOR_mask(
            norm_fx.numpy(), norm_fy.numpy(), pa.im_h, pa.im_w, pa.IOR_radius
        )
        action_mask = torch.logical_or(action_mask, new_mask)

    stop_flags = torch.stack(stop_flags, dim=1)
    # Truncate at terminal action
    trajs = []
    for i in range(normalized_fixs.size(0)):
        is_terminal = stop_flags[i] > 0.5
        if is_terminal.sum() == 0:
            ind = normalized_fixs.size(1)
        else:
            ind = is_terminal.to(torch.int8).argmax().item() + 1
        trajs.append(normalized_fixs[i, :ind])

    nonstop_trajs = [normalized_fixs[i] for i in range(normalized_fixs.size(0))]
    return trajs, nonstop_trajs


def actions2scanpaths(norm_fixs, patch_num, im_h, im_w):
    # convert actions to scanpaths
    scanpaths = []
    for traj in norm_fixs:
        task_name, img_name, condition, fixs = traj
        fixs = fixs.numpy()
        scanpaths.append(
            {
                "X": fixs[:, 0] * im_w,
                "Y": fixs[:, 1] * im_h,
                "name": img_name,
                "task": task_name,
                "condition": condition,
            }
        )
    return scanpaths


def compute_multilabel_metrics(logits, targets, threshold=0.5):
    """
    Calculate metrics for multilabel classification.

    Args:
        logits: Model predictions (before sigmoid) with shape L,B,D or B,D
        targets: Ground truth labels with same shape as logits
        threshold: Threshold to convert probabilities to binary predictions (default: 0.5)

    Returns:
        dict: Dictionary containing accuracy, F1, precision, and recall metrics
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Convert to binary predictions using threshold
    preds = (probs >= threshold).astype(np.int64)
    targets = (targets >= threshold).astype(np.int64)
    targets = targets.reshape(-1)
    preds = preds.reshape(-1)

    # Reshape to 2D if needed (samples, classes)
    if len(preds.shape) == 3:  # L,B,D
        preds = preds.reshape(-1, preds.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "f1_score": f1_score(targets, preds, average="macro", zero_division=0),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
    }

    # Add micro-averaged metrics
    metrics.update(
        {
            "f1_score_micro": f1_score(
                targets, preds, average="micro", zero_division=0
            ),
            "precision_micro": precision_score(
                targets, preds, average="micro", zero_division=0
            ),
            "recall_micro": recall_score(
                targets, preds, average="micro", zero_division=0
            ),
        }
    )

    return metrics


def evaluate(
    model,
    device,
    valid_img_loader,
    gazeloader,
    pa,
    log_dir=None,
):
    print(
        "Eval on {} batches of images and {} batches of fixations".format(
            len(valid_img_loader), len(gazeloader)
        )
    )
    model.eval()
    metrics_dict = {}
    all_logits = []
    all_targets = []
    for batch in tqdm(gazeloader, desc="computing eval scores"):
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
        targets = batch["label_findings"].detach().cpu()

        all_logits.append(logits)
        all_targets.append(targets)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics_dict = compute_multilabel_metrics(all_logits, all_targets)

    return metrics_dict
