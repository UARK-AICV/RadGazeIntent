import numpy as np
from torchvision import transforms

from .data import (
    FFN_IRL,
    SPTrans_Human_Gaze,
)
from .utils import compute_search_cdf, preprocess_fixations


def process_data(
    target_trajs,
    dataset_root,
    hparams,
    target_trajs_all,
    is_testing=False,
):
    print(
        "using",
        hparams.Train.repr,
        "dataset:",
        hparams.Data.name,
        "TAP:",
        hparams.Data.TAP,
    )

    # Rescale fixations and images if necessary
    ori_h, ori_w = 224, 224
    rescale_flag = hparams.Data.im_h != ori_h
    if rescale_flag:
        print(
            f"Rescaling image and fixation to {hparams.Data.im_h}x{hparams.Data.im_w}"
        )
        size = (hparams.Data.im_h, hparams.Data.im_w)
        ratio_h = hparams.Data.im_h / ori_h
        ratio_w = hparams.Data.im_w / ori_w
        for traj in target_trajs_all:
            traj["X"] = np.array(traj["X"]) * ratio_w
            traj["Y"] = np.array(traj["Y"]) * ratio_h
            traj["rescaled"] = True

    size = (hparams.Data.im_h, hparams.Data.im_w)
    transform_train = transforms.Compose(
        [
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_target_trajs_all = list(
        filter(lambda x: x["split"] == "test", target_trajs_all)
    )

    target_init_fixs = {}
    for traj in target_trajs_all:
        key = traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
        target_init_fixs[key] = (0.5, 0.5)

    cat_names = [
        "pleural_effusion",
        "pleural_other",
        "pneumonia",
        "fracture",
        "consolidation",
        "pneumothorax",
        "lung_lesion",
        "edema",
        "enlarged_cardiomediastinum",
        "atelectasis",
        "support_devices",
        "lung_opacity",
        "cardiomegaly",
    ]
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    human_mean_cdf = None
    if is_testing:
        # testing fixation data
        test_target_trajs = list(filter(lambda x: x["split"] == "test", target_trajs))
        assert len(test_target_trajs) > 0, "no testing data found!"
        test_task_img_pair = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in test_target_trajs
            ]
        )

        # print statistics
        traj_lens = list(map(lambda x: x["length"], test_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print(
            "average train scanpath length : {:.3f} (+/-{:.3f})".format(
                avg_traj_len, std_traj_len
            )
        )
        print("num of train trajs = {}".format(len(test_target_trajs)))

        if hparams.Data.TAP == "TP":
            human_mean_cdf, _ = compute_search_cdf(
                test_target_trajs, hparams.Data.max_traj_length
            )
            print("target fixation prob (test).:", human_mean_cdf)

        # load image data

        test_img_dataset = FFN_IRL(
            dataset_root,
            target_init_fixs,
            test_task_img_pair,
            transform_test,
            hparams.Data,
            catIds,
        )

        return {
            "catIds": catIds,
            "img_test": test_img_dataset,
            "gt_scanpaths": test_target_trajs,
        }

    else:
        # training fixation data
        train_target_trajs = list(filter(lambda x: x["split"] == "train", target_trajs))
        # print statistics
        traj_lens = list(map(lambda x: x["length"], train_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print(
            "average train scanpath length : {:.3f} (+/-{:.3f})".format(
                avg_traj_len, std_traj_len
            )
        )
        print("num of train trajs = {}".format(len(train_target_trajs)))

        train_task_img_pair = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in train_target_trajs
            ]
        )
        train_fix_labels = preprocess_fixations(
            train_target_trajs,
            catIds,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
        )

        # validation fixation data
        valid_target_trajs = list(filter(lambda x: x["split"] == "test", target_trajs))
        # print statistics
        traj_lens = list(map(lambda x: x["length"], valid_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print(
            "average valid scanpath length : {:.3f} (+/-{:.3f})".format(
                avg_traj_len, std_traj_len
            )
        )
        print("num of valid trajs = {}".format(len(valid_target_trajs)))

        if hparams.Data.TAP in ["TP", "TAP"]:
            tp_trajs = list(
                filter(
                    lambda x: x["condition"] == "present" and x["split"] == "test",
                    target_trajs_all,
                )
            )
            human_mean_cdf, _ = compute_search_cdf(
                tp_trajs, hparams.Data.max_traj_length
            )
            print("target fixation prob (valid).:", human_mean_cdf)

        valid_fix_labels = preprocess_fixations(
            valid_target_trajs,
            catIds,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
        )
        valid_target_trajs_TP = list(
            filter(lambda x: x["condition"] == "present", valid_target_trajs_all)
        )
        valid_fix_labels_TP = preprocess_fixations(
            valid_target_trajs_TP,
            catIds,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            discretize_fix=hparams.Data.discretize_fix,
        )

        valid_task_img_pair_TP = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in valid_target_trajs_all
                if traj["condition"] == "present"
            ]
        )

        valid_task_img_pair_all = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in valid_target_trajs_all
            ]
        )

        # load image data
        train_img_dataset = FFN_IRL(
            dataset_root,
            None,
            train_task_img_pair,
            transform_train,
            hparams.Data,
            catIds,
        )
        valid_img_dataset_all = FFN_IRL(
            dataset_root,
            None,
            valid_task_img_pair_all,
            transform_test,
            hparams.Data,
            catIds,
        )
        valid_img_dataset_TP = FFN_IRL(
            dataset_root,
            None,
            valid_task_img_pair_TP,
            transform_test,
            hparams.Data,
            catIds,
        )

        gaze_dataset_func = SPTrans_Human_Gaze

        train_HG_dataset = gaze_dataset_func(
            dataset_root,
            train_fix_labels,
            hparams.Data,
            transform_train,
            catIds,
            blur_action=True,
        )
        valid_HG_dataset = gaze_dataset_func(
            dataset_root,
            valid_fix_labels,
            hparams.Data,
            transform_test,
            catIds,
            blur_action=True,
        )
        valid_HG_dataset_TP = gaze_dataset_func(
            dataset_root,
            valid_fix_labels_TP,
            hparams.Data,
            transform_test,
            catIds,
            blur_action=True,
        )

        print(
            "num of training and eval fixations = {}, {}".format(
                len(train_HG_dataset), len(valid_HG_dataset)
            )
        )
        print(
            "num of training and eval images = {}, {} (TP)".format(
                len(train_img_dataset),
                len(valid_img_dataset_TP),
            )
        )

        return {
            "catIds": catIds,
            "img_train": train_img_dataset,
            "img_valid_TP": valid_img_dataset_TP,
            "img_valid": valid_img_dataset_all,
            "gaze_train": train_HG_dataset,
            "gaze_valid": valid_HG_dataset,
            "gaze_valid_TP": valid_HG_dataset_TP,
            "valid_scanpaths": valid_target_trajs_all,
            "human_cdf": human_mean_cdf,
        }
