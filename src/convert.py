# https://www.kaggle.com/datasets/girish17019/mobile-phone-defect-segmentation-dataset

import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:

    # project_name = "Mobile Phone Defect"
    dataset_path = "/home/grokhi/rawdata/mobile-phone-defect-segmentation"
    ds_name = "ds"
    batch_size = 30
    data_to_masks = {"oil": "ground_truth_2", "scratch": "ground_truth_1", "stain": "ground_truth_1", "good": "good"}

    images_ext = ".jpg"
    masks_ext = ".png"


    def get_unique_colors(img):
        unique_colors = []
        img = img.astype(np.int32)
        h, w = img.shape[:2]
        colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
        unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
        indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
        for col, indx in col2indx.items():
            if col != 0:
                unique_colors.append((col // (256**2), (col // 256) % 256, col % 256))

        return unique_colors


    def create_ann(image_path, subfolder):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        if subfolder == "good":
            return sly.Annotation(img_size=(img_height, img_wight), img_tags=[sly.Tag(good_tag)])

        mask_path = os.path.join(masks_path, get_file_name(image_path) + masks_ext)
        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)
            unique_colors = get_unique_colors(mask_np)
            for color in unique_colors:
                mask = np.all(mask_np == color, axis=2)
                ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
                for i in range(1, ret):
                    obj_mask = curr_mask == i
                    bitmap = sly.Bitmap(data=obj_mask)
                    if bitmap.area > 10:
                        obj_class = color_to_obj_class[color]
                        label = sly.Label(bitmap, obj_class)
                        labels.append(label)
        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    obj_class_oil = sly.ObjClass("oil", sly.Bitmap, color=(128, 0, 0))
    obj_class_scratch = sly.ObjClass("scratch", sly.Bitmap, color=(128, 128, 0))
    obj_class_stain = sly.ObjClass("stain", sly.Bitmap, color=(0, 128, 0))
    
    good_tag = sly.TagMeta('good', sly.TagValueType.NONE)

    color_to_obj_class = {
        (128, 0, 0): obj_class_oil,
        (128, 128, 0): obj_class_scratch,
        (0, 128, 0): obj_class_stain,
    }


    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class_oil, obj_class_scratch, obj_class_stain], tag_metas=[good_tag])
    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    for subfolder in ["good", "oil", "scratch", "stain"]:
        curr_images_path = os.path.join(dataset_path, subfolder)
        images_names = os.listdir(curr_images_path)

        progress = sly.Progress("Add {} data".format(subfolder), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(curr_images_path, image_name) for image_name in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]


            masks_path = os.path.join(dataset_path, data_to_masks[subfolder])
            anns_batch = [create_ann(image_path, subfolder) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))
    return project


