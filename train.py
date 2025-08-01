import argparse
import datetime
import os
import random
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.linalg as LA
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.config import JsonConfig
from common.utils import (
    transform_fixations,
)
from radgazeintent.builder import build
from radgazeintent.evaluation import evaluate

# from common.sinkhorn import SinkhornDistance

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


def compute_loss(model, batch, losses, loss_funcs, pa):
    img = batch["true_state"].to(device)
    inp_seq, inp_seq_high = transform_fixations(
        batch["normalized_fixations"],
        batch["is_padding"],
        hparams.Data,
        False,
        return_highres=True,
    )
    inp_seq = inp_seq.to(device)
    inp_padding_mask = inp_seq == pa.pad_idx
    logits = model(
        img,
        inp_seq,
        inp_padding_mask,
        inp_seq_high.to(device),
    )
    label_findings = batch["label_findings"].to(device)

    loss_dict = {}
    loss_dict["CE_segment"] = torch.nn.BCEWithLogitsLoss()(
        logits["outputs_logit"], label_findings
    )
    loss_dict["CE_segment"] += loss_funcs["CE_segment"](
        logits["outputs_logit"], label_findings, 13, True
    )
    return loss_dict


def train_iter(model, optimizer, batch, losses, loss_weights, loss_funcs, pa):
    assert len(losses) > 0, "no loss func assigned!"
    model.train()
    optimizer.zero_grad()

    loss_dict = compute_loss(model, batch, losses, loss_funcs, pa)
    loss = 0
    for k, v in loss_dict.items():
        loss += v * loss_weights[k]
    loss.backward()
    optimizer.step()

    for k in loss_dict:
        loss_dict[k] = loss_dict[k].item()

    return loss_dict


def get_eval_loss(model, eval_dataloader, losses, loss_funcs, pa):
    with torch.no_grad():
        model.eval()
        num_batches = 0
        avg_loss_dict = defaultdict(lambda: 0)
        for batch in tqdm(eval_dataloader, desc="computing eval loss"):
            loss_dict = compute_loss(model, batch, losses, loss_funcs, pa)
            for k in loss_dict:
                avg_loss_dict[k] += loss_dict[k].item()
            num_batches += 1
        for k in avg_loss_dict:
            avg_loss_dict[k] /= num_batches
        return avg_loss_dict


def run_evaluation():
    # Perform evaluation
    rst_tp = evaluate(
        model,
        device,
        valid_img_loader_tp,
        valid_gaze_loader_tp,
        hparams_tp.Data,
        log_dir=log_dir,
    )
    print("TP:", rst_tp)

    return rst_tp


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Reference:
        Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels.
                Wang, Z. et. al. MICCAI 2023.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        inputs = einops.rearrange(inputs, "b l d -> b d l")
        targets = einops.rearrange(targets, "b l d -> b d l")
    else:
        inputs = inputs.flatten(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    difference = LA.vector_norm(inputs - targets, ord=1, dim=-1)
    numerator = denominator - difference
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss.mean()
    return loss.sum() / num_objects


if __name__ == "__main__":
    args = parse_args()
    hparams = JsonConfig(args.hparams)
    dir = os.path.dirname(args.hparams)
    hparams_tp = JsonConfig(os.path.join(dir, "coco_search18_dense_SSL_TP.json"))

    dataset_root = args.dataset_root
    if dataset_root[-1] == "/":
        dataset_root = dataset_root[:-1]
    device = torch.device(f"cuda:{args.gpu_id}")

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
    if args.eval_only:
        run_evaluation()
    else:
        writer = SummaryWriter(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("Log dir:", log_dir)
        log_folder_runs = "./runs/{}".format(log_dir.split("/")[-1])
        if not os.path.exists(log_folder_runs):
            os.system(f"mkdir -p {log_folder_runs}")

        # Write configuration file to the log dir
        hparams.dump(log_dir, "config.json")

        print_every = 20
        max_iters = hparams.Train.max_iters
        save_every = hparams.Train.checkpoint_every
        eval_every = hparams.Train.evaluate_every
        pad_idx = hparams.Data.pad_idx
        use_focal_loss = hparams.Train.use_focal_loss
        loss_funcs = {
            "CE_segment": dice_loss,
        }

        loss_weights = {
            "CE_segment": 1.0,
        }
        losses = hparams.Train.losses
        loss_dict_avg = dict(zip(losses, [0] * len(losses)))
        print("loss weights:", loss_weights)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=hparams.Train.lr_steps, gamma=0.1
        )

        s_epoch = int(global_step / len(train_gaze_loader))

        last_time = datetime.datetime.now()
        for i_epoch in range(s_epoch, int(1e5)):
            scheduler.step()
            for i_batch, batch in enumerate(train_gaze_loader):
                loss_dict = train_iter(
                    model,
                    optimizer,
                    batch,
                    losses,
                    loss_weights,
                    loss_funcs,
                    hparams.Data,
                )
                for k in loss_dict:
                    loss_dict_avg[k] += loss_dict[k]

                if global_step % print_every == print_every - 1:
                    for k in loss_dict_avg:
                        loss_dict_avg[k] /= print_every

                    time = datetime.datetime.now()
                    eta = str(
                        (time - last_time) / print_every * (max_iters - global_step)
                    )
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, total loss: {:.3f}".format(
                        time[time.rfind(" ") + 1 : time.rfind(".")],
                        eta[: eta.rfind(".")],
                        global_step,
                        (global_step / max_iters) * 100,
                        i_epoch,
                        np.sum(list(loss_dict_avg.values())),
                    )

                    for k, v in loss_dict_avg.items():
                        log_msg += " {}_loss: {:.3f}".format(k, v)

                    print(log_msg)
                    log_dict(writer, loss_dict_avg, global_step, "train")
                    writer.add_scalar(
                        "train/lr", optimizer.param_groups[0]["lr"], global_step
                    )
                    for k in loss_dict_avg:
                        loss_dict_avg[k] = 0

                # Evaluate
                if global_step % eval_every == eval_every - 1:
                    rst_tp = run_evaluation()
                    if rst_tp is not None:
                        log_dict(writer, rst_tp, global_step, "eval_TP")
                    writer.add_scalar(
                        "train/epoch", global_step / len(train_gaze_loader), global_step
                    )
                    os.system(f"cp {log_dir}/events* {log_folder_runs}")

                if global_step % save_every == save_every - 1:
                    save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")
                    if isinstance(model, torch.nn.DataParallel):
                        model_weights = model.module.state_dict()
                    else:
                        model_weights = model.state_dict()
                    torch.save(
                        {
                            "model": model_weights,
                            "optimizer": optimizer.state_dict(),
                            "step": global_step + 1,
                        },
                        save_path,
                    )
                    print(f"Saved checkpoint to {save_path}.")
                global_step += 1
                if global_step >= max_iters:
                    print("Exit program!")
                    break
            else:
                continue
            break  # Break outer loop

        # Copy to log file to ./runs
        os.system(f"cp {log_dir}/events* {log_folder_runs}")
