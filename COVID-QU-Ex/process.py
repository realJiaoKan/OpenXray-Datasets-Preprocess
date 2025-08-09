import os
import glob
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np


def load_and_preprocess(data_dir, target_size=(64, 64)):
    categories = {"COVID-19": 0, "Non-COVID": 1, "Normal": 2}

    images = []
    masks = []
    labels = []
    label_names = list(categories.keys())

    for category, label in categories.items():
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Path {category_path} does not exist, skipping category {category}")
            continue

        image_files = sorted(glob.glob(os.path.join(category_path, "images", "*.png")))

        masks_dirnames = ["masks", "infection masks", "lung masks"]
        masks_dir = None
        for d in masks_dirnames:
            candidate = os.path.join(category_path, d)
            if os.path.isdir(candidate):
                masks_dir = candidate
                break
        if masks_dir is None:
            print(f"No masks folder found for {category_path}, skipping")
            continue
        masks_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

        for img_path, msk_path in tqdm(
            zip(image_files, masks_files),
            total=len(image_files),
            desc=f"Processing {category}",
        ):
            try:
                assert (
                    Path(img_path).stem == Path(msk_path).stem
                ), "Image and mask filenames do not match"
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                if img is None or msk is None:
                    print(f"Warning: Could not read pair {img_path} and {msk_path}")
                    continue

                img_resized = cv2.resize(img, target_size)
                msk_resized = cv2.resize(msk, target_size)
                img_normalized = img_resized.astype(np.float32) / 255.0
                msk_resized = msk_resized.astype(np.float32) / 255.0

                # Add a channel dimension before (H, W) to images and masks
                img_normalized = np.expand_dims(img_normalized, axis=0)
                msk_resized = np.expand_dims(msk_resized, axis=0)

                images.append(img_normalized)
                masks.append(msk_resized)
                labels.append(label)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

    return np.array(images), np.array(masks), np.array(labels), label_names


def create_dataset(train_dir, test_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img_train, msk_train, lbl_train, label_names = load_and_preprocess(train_dir)
    img_test, msk_test, lbl_test, _ = load_and_preprocess(test_dir)

    if len(img_train) > 0:
        print(f"Train set size: {len(img_train)}")
        print(f"Image size: {img_train[0].shape}")
    if len(img_test) > 0:
        print(f"Test set size: {len(img_test)}")

    train_path = os.path.join(output_dir, "train.npz")
    np.savez_compressed(
        train_path,
        images=img_train,
        masks=msk_train,
        labels=lbl_train,
        label_names=label_names,
    )

    test_path = os.path.join(output_dir, "test.npz")
    np.savez_compressed(
        test_path,
        images=img_test,
        masks=msk_test,
        labels=lbl_test,
        label_names=label_names,
    )

    return img_train, msk_train, lbl_train, img_test, msk_test, lbl_test, label_names


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(
        current_dir,
        "Data",
        "Raw",
        "Lung Segmentation Data",
        "Lung Segmentation Data",
    )
    train_dir = os.path.join(data_root, "Train")
    test_dir = os.path.join(data_root, "Test")
    output_dir = os.path.join(current_dir, "Data", "Processed")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(
            f"Error: Data source directories not found: \nTrain: {train_dir}\nTest: {test_dir}"
        )
        exit(1)

    create_dataset(train_dir, test_dir, output_dir)
