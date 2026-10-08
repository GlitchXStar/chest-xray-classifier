import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import IMG_SIZE, BATCH_SIZE, CLASSES, NUM_CLASSES


# ============================================================
# IMAGE INDEXING
# ============================================================

def index_images(base_path):
    """
    Walks all images_0xx subfolders under base_path and
    returns a dict {filename: full_path}.
    """
    image_path_dict = {}
    for folder in os.listdir(base_path):
        if folder.startswith('images_'):
            d = os.path.join(base_path, folder, 'images')
            if os.path.exists(d):
                for fname in os.listdir(d):
                    image_path_dict[fname] = os.path.join(d, fname)
    return image_path_dict


# ============================================================
# CSV LOADING & LABEL CREATION
# ============================================================

def load_dataframe(csv_path, image_path_dict):
    """
    Reads Data_Entry_2017.csv, creates one binary column per
    disease, and attaches resolved image paths.
    """
    df = pd.read_csv(csv_path)

    for cls in CLASSES:
        df[cls] = df['Finding Labels'].apply(
            lambda x: 1 if cls in x else 0
        )

    df['image_path'] = df['Image Index'].map(image_path_dict)
    df = df.dropna(subset=['image_path'])
    return df


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

def split_dataframe(df, val_size=0.10, test_size=0.10, seed=42):
    """
    Stratified split on 'No Finding' flag to preserve
    normal/disease ratio across all three sets.
    Returns train_df, val_df, test_df.
    """
    df['no_finding'] = (df['Finding Labels'] == 'No Finding').astype(int)

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df['no_finding'],
        random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df['no_finding'],
        random_state=seed
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image_path, labels):
    """
    Loads a JPEG, resizes to 320x320, applies DenseNet
    preprocessing (zero-centers per ImageNet stats).
    """
    raw   = tf.io.read_file(image_path)
    img   = tf.image.decode_jpeg(raw, channels=3)
    img   = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img   = tf.cast(img, tf.float32)
    img   = tf.keras.applications.densenet.preprocess_input(img)
    return img, labels


# ============================================================
# AUGMENTATION (training only)
# ============================================================

def augment(img, labels):
    """
    Light augmentation — only transforms that are clinically
    plausible for chest X-rays.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    # Small random rotation via crop-and-resize
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 20, IMG_SIZE + 20)
    img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])

    return img, labels


# ============================================================
# TF.DATA PIPELINE BUILDERS
# ============================================================

def build_dataset(df, training=False, batch_size=BATCH_SIZE):
    """
    Builds a tf.data.Dataset from a dataframe.
    Set training=True to enable augmentation and shuffling.
    """
    paths  = df['image_path'].values
    labels = df[CLASSES].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_all_datasets(base_path, csv_path,
                       val_size=0.10, test_size=0.10,
                       batch_size=BATCH_SIZE, seed=42):
    """
    Full pipeline: index → load CSV → split → build datasets.
    Returns train_ds, val_ds, test_ds, test_df.
    """
    image_path_dict      = index_images(base_path)
    df                   = load_dataframe(csv_path, image_path_dict)
    train_df, val_df, test_df = split_dataframe(
        df, val_size=val_size, test_size=test_size, seed=seed
    )

    train_ds = build_dataset(train_df, training=True,  batch_size=batch_size)
    val_ds   = build_dataset(val_df,   training=False, batch_size=batch_size)
    test_ds  = build_dataset(test_df,  training=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds, test_df