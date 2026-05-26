# =========================================================
# preprocessing/frame_sampling.py
# Frame sampling and Xception preprocessing utilities
# =========================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf


IMG_SIZE = (224, 224)
MAX_FRAMES = 30


def is_valid_image(path):
    """
    Check whether an image can be decoded.
    """
    try:
        img = tf.io.read_file(path)
        tf.io.decode_image(img, channels=3, expand_animations=False)
        return True
    except Exception:
        return False


def list_frame_files(video_dir):
    """
    List image frames inside one video folder.
    """
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    frames = [
        f for f in sorted(os.listdir(video_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    return frames


def sample_frame_paths(
    video_dir,
    max_frames=30,
    seed=42,
    sampling_mode="random"
):
    """
    Sample frame paths from a video folder.

    Parameters
    ----------
    video_dir : str
        Folder containing extracted frames for one video.
    max_frames : int
        Number of frames to sample.
    seed : int
        Random seed used when sampling_mode='random'.
    sampling_mode : str
        'random' or 'uniform'.

    Returns
    -------
    selected_paths : list
        Valid selected frame paths.
    invalid_paths : list
        Selected but invalid/corrupted frame paths.

    Notes
    -----
    The experiments report 30 frames per video. If using seeded frame-sampling
    runs, use sampling_mode='random'. If using deterministic temporal sampling,
    use sampling_mode='uniform'.
    """

    frames = list_frame_files(video_dir)

    if len(frames) == 0:
        return [], []

    if len(frames) <= max_frames:
        selected_frames = frames
    else:
        if sampling_mode == "random":
            rng = random.Random(seed)
            selected_frames = rng.sample(frames, max_frames)
            selected_frames = sorted(selected_frames)

        elif sampling_mode == "uniform":
            indices = np.linspace(
                0,
                len(frames) - 1,
                max_frames,
                dtype=int
            )
            selected_frames = [frames[i] for i in indices]

        else:
            raise ValueError("sampling_mode must be 'random' or 'uniform'.")

    selected_paths = []
    invalid_paths = []

    for frame_name in selected_frames:
        frame_path = os.path.join(video_dir, frame_name)

        if is_valid_image(frame_path):
            selected_paths.append(frame_path)
        else:
            invalid_paths.append(frame_path)

    return selected_paths, invalid_paths


def load_split_csv(split_csv, dataset_root):
    """
    Load video-level split file and convert relative paths to full paths.

    Expected CSV columns:
        dataset,video_id,relative_path,label,split,seed

    Parameters
    ----------
    split_csv : str
        Path to split CSV.
    dataset_root : str
        Root folder for the dataset.

    Returns
    -------
    df : pandas.DataFrame
        Split dataframe with an added full_path column.
    """

    df = pd.read_csv(split_csv)

    required_cols = {
        "dataset",
        "video_id",
        "relative_path",
        "label",
        "split"
    }

    missing = required_cols.difference(df.columns)

    if missing:
        raise KeyError(f"{split_csv} missing required columns: {missing}")

    df["full_path"] = df["relative_path"].apply(
        lambda p: os.path.join(dataset_root, p)
    )

    return df


def collect_sampled_frames_from_split(
    split_csv,
    dataset_root,
    target_split,
    max_frames=30,
    seed=42,
    sampling_mode="random"
):
    """
    Collect sampled frame paths and labels for one split.

    Example target_split:
        train, validation, heldout, finetune, evaluation
    """

    df = load_split_csv(split_csv, dataset_root)
    df_split = df[df["split"] == target_split].copy()

    all_paths = []
    all_labels = []
    all_video_ids = []
    invalid_rows = []

    for _, row in df_split.iterrows():
        video_dir = row["full_path"]
        label = int(row["label"])
        video_id = row["video_id"]

        frame_paths, invalid_paths = sample_frame_paths(
            video_dir=video_dir,
            max_frames=max_frames,
            seed=seed,
            sampling_mode=sampling_mode
        )

        all_paths.extend(frame_paths)
        all_labels.extend([label] * len(frame_paths))
        all_video_ids.extend([video_id] * len(frame_paths))

        for invalid_path in invalid_paths:
            invalid_rows.append({
                "video_id": video_id,
                "label": label,
                "invalid_frame_path": invalid_path
            })

    sampled_df = pd.DataFrame({
        "frame_path": all_paths,
        "video_id": all_video_ids,
        "label": all_labels
    })

    invalid_df = pd.DataFrame(invalid_rows)

    return sampled_df, invalid_df


def load_and_preprocess_xception(path, label):
    """
    Load and preprocess one image for Xception.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.xception.preprocess_input(img)

    return img, tf.cast(label, tf.float32)


def make_tf_dataset(paths, labels, batch_size=16, training=False, seed=42):
    """
    Build tf.data.Dataset from frame paths and labels.
    """

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_and_preprocess_xception,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
