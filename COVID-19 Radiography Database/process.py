import os
import glob
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess(data_dir, target_size=(256, 256)):
    categories = {"COVID": 0, "Lung_Opacity": 1, "Normal": 2, "Viral Pneumonia": 3}

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
        masks_files = sorted(glob.glob(os.path.join(category_path, "masks", "*.png")))

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
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue

                img_resized = cv2.resize(img, target_size)
                msk_resized = cv2.resize(msk, target_size)
                img_normalized = img_resized.astype(np.float32) / 255.0
                msk_resized = msk_resized.astype(np.float32) / 255.0

                images.append(img_normalized)
                masks.append(msk_resized)
                labels.append(label)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

    return np.array(images), np.array(masks), np.array(labels), label_names


def create_dataset(data_dir, output_dir, test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    images, masks, labels, label_names = load_and_preprocess(data_dir)

    print(f"Total number of images: {len(images)}")
    print(f"Image size: {images[0].shape}")
    for i, category in enumerate(label_names):
        count = np.sum(labels == i)
        print(f"{category}: {count} images")

    X_train, X_test, M_train, M_test, y_train, y_test = train_test_split(
        images,
        masks,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,  # Keep class distribution in train/test split
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    train_path = os.path.join(output_dir, "train.npz")
    print(f"Saving train set to: {train_path}")
    np.savez_compressed(
        train_path,
        images=X_train,
        masks=M_train,
        labels=y_train,
        label_names=label_names,
    )

    test_path = os.path.join(output_dir, "test.npz")
    print(f"Saving test set to: {test_path}")
    np.savez_compressed(
        test_path, images=X_test, masks=M_test, labels=y_test, label_names=label_names
    )


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "Data", "Raw")
    output_dir = os.path.join(current_dir, "Data", "Processed")

    if not os.path.exists(data_dir):
        print(f"Error: Data source directory {data_dir} does not exist")
        exit(1)

    create_dataset(data_dir, output_dir)
