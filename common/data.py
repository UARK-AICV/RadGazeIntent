from os.path import join

import numpy as np
import scipy.ndimage as filters
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from . import utils


class RolloutStorage(object):
    def __init__(self, trajs_all, shuffle=True, norm_adv=False):
        self.is_composite_state = isinstance(trajs_all[0]["curr_states"], list)
        if self.is_composite_state:
            num_state_comp = len(trajs_all[0]["curr_states"])
            self.obs_fovs = [
                torch.cat([traj["curr_states"][i] for traj in trajs_all])
                for i in range(num_state_comp)
            ]
        else:
            self.obs_fovs = [torch.cat([traj["curr_states"] for traj in trajs_all])]
        # self.obs_fovs = torch.cat([traj["curr_states"] for traj in trajs_all])
        self.actions = torch.cat([traj["actions"] for traj in trajs_all])
        self.lprobs = torch.cat([traj["log_probs"] for traj in trajs_all])
        self.tids = torch.cat([traj["task_id"] for traj in trajs_all])
        self.returns = torch.cat([traj["acc_rewards"] for traj in trajs_all]).view(-1)
        self.advs = torch.cat([traj["advantages"] for traj in trajs_all]).view(-1)
        if norm_adv:
            self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)
        self.is_zero_shot = trajs_all[0]["hr_feats"] is not None
        if self.is_zero_shot:
            self.hr_feats = torch.cat([traj["hr_feats"] for traj in trajs_all])

        self.sample_num = self.actions.size(0)
        self.shuffle = shuffle

    def get_generator(self, minibatch_size):
        minibatch_size = min(self.sample_num, minibatch_size)
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.sample_num)), minibatch_size, drop_last=True
        )
        for ind in sampler:
            obs_fov_batch = [obs_fovs[ind] for obs_fovs in self.obs_fovs]
            actions_batch = self.actions[ind]
            tids_batch = self.tids[ind]
            return_batch = self.returns[ind]
            log_probs_batch = self.lprobs[ind]
            advantage_batch = self.advs[ind]

            if self.is_zero_shot:
                hr_batch = self.hr_feats[ind]
                yield (
                    (*obs_fov_batch, tids_batch, hr_batch),
                    actions_batch,
                    return_batch,
                    log_probs_batch,
                    advantage_batch,
                )
            else:
                yield (
                    (*obs_fov_batch, tids_batch),
                    actions_batch,
                    return_batch,
                    log_probs_batch,
                    advantage_batch,
                )


class FakeDataRollout(object):
    def __init__(self, trajs_all, minibatch_size, shuffle=True):
        self.is_composite_state = isinstance(trajs_all[0]["curr_states"], list)
        if self.is_composite_state:
            num_state_comp = len(trajs_all[0]["curr_states"])
            self.GS = [
                torch.cat([traj["curr_states"][i] for traj in trajs_all])
                for i in range(num_state_comp)
            ]
        else:
            self.GS = [torch.cat([traj["curr_states"] for traj in trajs_all])]
        self.GA = torch.cat([traj["actions"] for traj in trajs_all]).unsqueeze(1)
        self.tids = torch.cat([traj["task_id"] for traj in trajs_all])
        self.GP = torch.exp(
            torch.cat([traj["log_probs"] for traj in trajs_all])
        ).unsqueeze(1)
        self.is_zero_shot = trajs_all[0]["hr_feats"] is not None
        if self.is_zero_shot:
            self.hr_feats = torch.cat([traj["hr_feats"] for traj in trajs_all])
        self.sample_num = self.GA.size(0)
        self.shuffle = shuffle
        self.batch_size = min(minibatch_size, self.sample_num)

    def __len__(self):
        return int(self.sample_num // self.batch_size)

    def get_generator(self):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.sample_num)), self.batch_size, drop_last=True
        )
        for ind in sampler:
            GS_batch = [GS[ind] for GS in self.GS]
            tid_batch = self.tids[ind]
            GA_batch = self.GA[ind]
            GP_batch = self.GP[ind]

            if self.is_zero_shot:
                hr_batch = self.hr_feats[ind]
                yield (*GS_batch, GA_batch, tid_batch), hr_batch, GP_batch
            else:
                yield (*GS_batch, GA_batch, tid_batch), GP_batch


def get_coco_annos_by_img_name(coco_annos, img_name):
    img_id = int(img_name[:-4])
    # List: get annotation id from coco
    coco_annos_train, coco_annos_val = coco_annos
    ann_ids = coco_annos_train.getAnnIds(imgIds=img_id)
    coco = coco_annos_train
    if len(ann_ids) == 0:
        ann_ids = coco_annos_val.getAnnIds(imgIds=img_id)
        coco = coco_annos_val

    # Dictionary: target coco_annotation file for an image
    coco_annotation = coco.loadAnns(ann_ids)

    # number of objects in the image
    num_objs = len(coco_annotation)

    # Bounding boxes for objects
    # In coco format, bbox = [xmin, ymin, width, height]
    # In pytorch, the input should be [xmin, ymin, xmax, ymax]
    boxes = []
    for i in range(num_objs):
        xmin = coco_annotation[i]["bbox"][0]
        ymin = coco_annotation[i]["bbox"][1]
        xmax = xmin + coco_annotation[i]["bbox"][2]
        ymax = ymin + coco_annotation[i]["bbox"][3]
        boxes.append([xmin, ymin, xmax, ymax])
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # Labels (In my case, I only one class: target class or background)
    labels = torch.ones((num_objs,), dtype=torch.int64)
    # Tensorise img_id
    img_id = torch.tensor([img_id])
    # Size of bbox (Rectangular)
    areas = []
    for i in range(num_objs):
        areas.append(coco_annotation[i]["area"])
    areas = torch.as_tensor(areas, dtype=torch.float32)
    # Iscrowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    # Annotation is in dictionary format
    my_annotation = {}
    my_annotation["boxes"] = boxes
    my_annotation["labels"] = labels
    my_annotation["image_id"] = img_id
    my_annotation["area"] = areas
    my_annotation["iscrowd"] = iscrowd
    return my_annotation


class FFN_IRL(Dataset):
    def __init__(
        self,
        root_dir,
        initial_fix,
        img_info,
        transform,
        pa,
        catIds,
        coco_annos=None,
    ):
        self.img_info = img_info
        self.root_dir = root_dir
        self.img_dir = join(root_dir, "../images")
        self.transform = transform
        self.pa = pa
        self.initial_fix = initial_fix
        self.catIds = catIds
        self.fv_tid = 0 if self.pa.TAP == "FV" else len(self.catIds)

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        imgId = self.img_info[idx]
        cat_name, img_name, condition = imgId.split("*")

        c = cat_name.replace(" ", "_")
        im_path = "{}/{}".format(self.img_dir, img_name)
        im = Image.open(im_path)
        im_tensor = self.transform(im)

        coding = torch.zeros(1, self.pa.patch_count)

        # create action mask
        action_mask = np.zeros(
            self.pa.patch_num[0] * self.pa.patch_num[1], dtype=np.uint8
        )

        ret = {
            "img_name": img_name,
            "cat_name": cat_name,
            "im_tensor": im_tensor,
            "label_coding": coding,
            "condition": condition,
            "action_mask": torch.from_numpy(action_mask),
        }

        return ret


class SPTrans_Human_Gaze(Dataset):
    """
    Human gaze data for two-pathway dense transformer
    """

    def __init__(
        self,
        root_dir,
        fix_labels,
        pa,
        transform,
        catIds,
        blur_action=False,
        acc_foveal=True,
    ):
        self.root_dir = root_dir
        self.img_dir = join(root_dir, "../images")
        self.pa = pa
        self.transform = transform
        self.to_tensor = T.ToTensor()

        # Remove fixations longer than max_traj_length
        self.fix_labels = list(
            filter(lambda x: len(x[3]) <= pa.max_traj_length, fix_labels)
        )

        self.catIds = catIds
        self.blur_action = blur_action
        self.acc_foveal = acc_foveal

        self.resize = T.Resize(size=(pa.im_h // 2, pa.im_w // 2))
        self.resize2 = T.Resize(size=(pa.im_h // 4, pa.im_w // 4))

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        (
            img_name,
            cat_name,
            condition,
            fixs,
            label_findings,
            action,
            is_last,
            sid,
            dura,
        ) = self.fix_labels[idx]
        im_path = "{}/{}".format(self.img_dir, img_name)
        im = Image.open(im_path).convert("RGB")

        im_tensor = self.transform(im.copy())
        assert (
            im_tensor.shape[-1] == self.pa.im_w and im_tensor.shape[-2] == self.pa.im_h
        ), "wrong image size."

        IOR_weight_map = np.zeros((self.pa.im_h, self.pa.im_w), dtype=np.float32)

        IOR_weight_map += 1  # Set base weight to 1

        scanpath_length = len(fixs)
        if scanpath_length == 0:
            fixs = [(0, 0)]
        # Pad fixations to max_traj_lenght
        fixs = fixs + [fixs[-1]] * (self.pa.max_traj_length - len(fixs))
        is_padding = torch.zeros(self.pa.max_traj_length)
        is_padding[scanpath_length:] = 1
        # Pad finding label as well.
        label_findings = label_findings + [label_findings[-1]] * (
            self.pa.max_traj_length - len(label_findings)
        )

        fixs_tensor = torch.FloatTensor(fixs)
        # Normalize to 0-1 (avoid 1 by adding 1 pixel).
        fixs_tensor /= torch.FloatTensor([self.pa.im_w + 1, self.pa.im_h + 1])

        next_fixs_tensor = fixs_tensor.clone()
        if not is_last:
            x, y = utils.action_to_pos(action, [1, 1], [self.pa.im_w, self.pa.im_h])
            next_fix = torch.FloatTensor([x, y]) / torch.FloatTensor(
                [self.pa.im_w, self.pa.im_h]
            )
            next_fixs_tensor[scanpath_length:] = next_fix

        target_fix_map = np.zeros(self.pa.im_w * self.pa.im_h, dtype=np.float32)
        if not is_last:
            target_fix_map[action] = 1
            target_fix_map = target_fix_map.reshape(self.pa.im_h, -1)
            target_fix_map = filters.gaussian_filter(
                target_fix_map, sigma=self.pa.target_fix_map_sigma
            )
            target_fix_map /= target_fix_map.max()  # Normalize peak value to 1
        else:
            target_fix_map = target_fix_map.reshape(self.pa.im_h, -1)

        ret = {
            "task_id": 0,
            "true_state": im_tensor,
            "target_fix_map": target_fix_map,
            "true_action": torch.tensor([action], dtype=torch.long),
            "img_name": img_name,
            "task_name": cat_name,
            "label_findings": torch.tensor(label_findings, dtype=torch.float),
            "normalized_fixations": fixs_tensor,
            "next_normalized_fixations": next_fixs_tensor,
            "is_TP": condition == "present",
            "is_last": is_last,
            "is_padding": is_padding,
            "true_or_fake": 1.0,
            "IOR_weight_map": IOR_weight_map,
            "scanpath_length": scanpath_length,
            "duration": dura,
            "subj_id": sid - 1,  # sid ranges from [1, 10]
        }

        return ret
