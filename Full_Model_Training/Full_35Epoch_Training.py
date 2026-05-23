# =========================================================
# FINAL 35-EPOCH TRAINING USING SELECTED BEST HYPERPARAMETERS
# =========================================================

import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
MAX_FRAMES = 30
FULL_EPOCHS = 35
WARMUP_EPOCHS = 5
SEEDS = [42, 123]

SOURCE_REAL_VIDEOS = 400
SOURCE_FAKE_VIDEOS = 400

DATA_REAL = "/kaggle/input/datasets/bnjyothi/celebdf/celeb_df_V2_real_frames/celeb_df_V2_real_frames"
DATA_FAKE = "/kaggle/input/datasets/bnjyothi/celebdf/celeb_df_V2_fake_frames/celeb_df_V2_fake_frames"

BEST_BOAUC_PATH = "best_boauc_params.json"
BEST_AABO_PATH = "best_aabo_params.json"

OUTPUT_DIR = "final_35epoch_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# RANDOM SEED
# =========================================================

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================================================
# DATA HELPERS
# =========================================================

def is_valid_jpeg(path):
    try:
        img = tf.io.read_file(path)
        tf.io.decode_jpeg(img, channels=3)
        return True
    except Exception:
        return False


def collect_video_dirs(folder, label):
    vids = []

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            vids.append((path, label))

    return vids


def select_video_subset(real_videos, fake_videos, n_real, n_fake, seed=42):
    rng = np.random.default_rng(seed)

    if len(real_videos) < n_real:
        raise ValueError(f"Only {len(real_videos)} real videos found, but {n_real} requested.")

    if len(fake_videos) < n_fake:
        raise ValueError(f"Only {len(fake_videos)} fake videos found, but {n_fake} requested.")

    real_idx = rng.choice(len(real_videos), n_real, replace=False)
    fake_idx = rng.choice(len(fake_videos), n_fake, replace=False)

    selected_real = [real_videos[i] for i in real_idx]
    selected_fake = [fake_videos[i] for i in fake_idx]

    selected = selected_real + selected_fake
    rng.shuffle(selected)

    return selected


def sample_frames(video_dirs, labels, max_frames=30, seed=42):
    rng = random.Random(seed)

    paths = []
    y = []

    for vid, label in zip(video_dirs, labels):
        frames = sorted(os.listdir(vid))
        rng.shuffle(frames)
        frames = frames[:max_frames]

        for f in frames:
            full = os.path.join(vid, f)

            if is_valid_jpeg(full):
                paths.append(full)
                y.append(label)

    return paths, y


def balance(paths, labels, seed=42):
    rng = np.random.default_rng(seed)

    paths = np.array(paths)
    labels = np.array(labels)

    real_idx = np.where(labels == 0)[0]
    fake_idx = np.where(labels == 1)[0]

    n = min(len(real_idx), len(fake_idx))

    real_idx = rng.choice(real_idx, n, replace=False)
    fake_idx = rng.choice(fake_idx, n, replace=False)

    idx = np.concatenate([real_idx, fake_idx])
    rng.shuffle(idx)

    return paths[idx].tolist(), labels[idx].tolist()


def load_and_preprocess_eval(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.xception.preprocess_input(img)

    return img, tf.cast(label, tf.float32)


def load_and_preprocess_train(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.xception.preprocess_input(img)

    return img, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(4000, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(load_and_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(load_and_preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =========================================================
# LOAD DATA USING VIDEO-LEVEL SPLIT
# =========================================================

def prepare_data():
    real_videos = collect_video_dirs(DATA_REAL, 0)
    fake_videos = collect_video_dirs(DATA_FAKE, 1)

    selected_videos = select_video_subset(
        real_videos=real_videos,
        fake_videos=fake_videos,
        n_real=SOURCE_REAL_VIDEOS,
        n_fake=SOURCE_FAKE_VIDEOS,
        seed=42
    )

    video_paths = [v[0] for v in selected_videos]
    video_labels = [v[1] for v in selected_videos]

    train_videos, val_videos, train_labels, val_labels = train_test_split(
        video_paths,
        video_labels,
        stratify=video_labels,
        test_size=0.2,
        random_state=42
    )

    train_paths_raw, train_y_raw = sample_frames(
        train_videos,
        train_labels,
        max_frames=MAX_FRAMES,
        seed=42
    )

    val_paths, val_y = sample_frames(
        val_videos,
        val_labels,
        max_frames=MAX_FRAMES,
        seed=42
    )

    print("\n===== DATA SUMMARY =====")
    print(f"Selected videos: {len(selected_videos)}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    print(f"Training frames: {len(train_paths_raw)}")
    print(f"Validation frames: {len(val_paths)}")

    return train_paths_raw, train_y_raw, val_paths, val_y


# =========================================================
# DWAF MODEL
# =========================================================

class XceptionDWAF(tf.keras.Model):
    def __init__(self, heads=4, key_dim=32, momentum=0.5, alpha_scale=1.0):
        super().__init__()

        self.heads = heads
        self.momentum = momentum

        self.base = tf.keras.applications.Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        self.base.trainable = False

        self.reshape = tf.keras.layers.Reshape((49, 2048))

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=key_dim
        )

        if 2048 % heads != 0:
            raise ValueError("2048 must be divisible by the number of attention heads.")

        dim = 2048 // heads

        self.real_ema = [
            tf.Variable(tf.zeros(dim), trainable=False)
            for _ in range(heads)
        ]

        self.fake_ema = [
            tf.Variable(tf.zeros(dim), trainable=False)
            for _ in range(heads)
        ]

        self.alpha = tf.Variable(
            alpha_scale * tf.ones(heads),
            trainable=True
        )

        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x, y=None, training=False, epoch=0):
        x = self.base(x, training=training)
        x = self.reshape(x)

        attn = self.mha(x, x)

        heads = tf.split(attn, self.heads, axis=-1)

        pooled = [tf.reduce_mean(h, axis=1) for h in heads]
        pooled = [tf.math.l2_normalize(h, axis=-1) for h in pooled]

        deltas = []

        if training and y is not None:
            y = tf.cast(y, tf.int32)

            for i, h in enumerate(pooled):
                real_samples = tf.boolean_mask(h, y == 0)
                fake_samples = tf.boolean_mask(h, y == 1)

                if tf.shape(real_samples)[0] > 0:
                    real_mean = tf.reduce_mean(real_samples, axis=0)
                    self.real_ema[i].assign(
                        self.momentum * self.real_ema[i]
                        + (1.0 - self.momentum) * real_mean
                    )

                if tf.shape(fake_samples)[0] > 0:
                    fake_mean = tf.reduce_mean(fake_samples, axis=0)
                    self.fake_ema[i].assign(
                        self.momentum * self.fake_ema[i]
                        + (1.0 - self.momentum) * fake_mean
                    )

                real_norm = tf.norm(self.real_ema[i])
                fake_norm = tf.norm(self.fake_ema[i])

                delta = (real_norm - fake_norm) / (real_norm + fake_norm + 1e-6)
                delta = tf.clip_by_value(delta, -0.5, 0.5)

                deltas.append(delta)

        else:
            for i in range(self.heads):
                real_norm = tf.norm(self.real_ema[i])
                fake_norm = tf.norm(self.fake_ema[i])

                delta = (real_norm - fake_norm) / (real_norm + fake_norm + 1e-6)
                delta = tf.clip_by_value(delta, -0.5, 0.5)

                deltas.append(delta)

        deltas = tf.stack(deltas)

        if epoch < WARMUP_EPOCHS:
            weights = tf.ones_like(deltas) / tf.cast(self.heads, deltas.dtype)
        else:
            weights = tf.nn.softmax(self.alpha * deltas)

        fused = tf.add_n(
            [w * f for w, f in zip(tf.unstack(weights), pooled)]
        )

        out = self.classifier(fused)

        return out, deltas, weights


# =========================================================
# LOAD BEST PARAMETERS
# =========================================================

def load_best_params(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run hyperparameter search first."
        )

    with open(path, "r") as f:
        params = json.load(f)

    required_keys = ["lr", "heads", "key_dim", "momentum", "alpha_scale"]

    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing key in {path}: {key}")

    return params


# =========================================================
# FINAL 35-EPOCH TRAINING
# =========================================================

def run_final_training(best_hparams, method_name, seed,
                       train_paths_raw, train_y_raw,
                       val_paths, val_y):

    set_seed(seed)

    print("\n====================================")
    print(f"Final 35-epoch training | {method_name} | Seed {seed}")
    print("====================================")
    print("Selected hyperparameters:")
    for k, v in best_hparams.items():
        print(f"{k}: {v}")

    train_paths, train_y = balance(
        train_paths_raw,
        train_y_raw,
        seed=seed
    )

    train_ds = make_dataset(
        train_paths,
        train_y,
        training=True,
        seed=seed
    )

    val_ds_final = make_dataset(
        val_paths,
        val_y,
        training=False,
        seed=seed
    )

    model = XceptionDWAF(
        heads=best_hparams["heads"],
        key_dim=best_hparams["key_dim"],
        momentum=best_hparams["momentum"],
        alpha_scale=best_hparams["alpha_scale"]
    )

    optimizer = tf.keras.optimizers.Adam(best_hparams["lr"])
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    train_loss_history = []
    val_loss_history = []
    val_auc_history = []

    final_deltas = None
    final_weights = None
    all_y = None
    all_probs = None

    for epoch in range(FULL_EPOCHS):
        epoch_train_losses = []

        for x_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                preds, deltas, weights = model(
                    x_batch,
                    y=y_batch,
                    training=True,
                    epoch=epoch
                )

                loss = loss_fn(y_batch, preds)

            grads = tape.gradient(loss, model.trainable_variables)

            grads_and_vars = [
                (g, v)
                for g, v in zip(grads, model.trainable_variables)
                if g is not None
            ]

            optimizer.apply_gradients(grads_and_vars)

            epoch_train_losses.append(loss.numpy())

        train_loss = float(np.mean(epoch_train_losses))
        train_loss_history.append(train_loss)

        all_y_epoch = []
        all_probs_epoch = []
        epoch_val_losses = []

        delta_batches = []
        weight_batches = []

        for x_val, y_val_batch in val_ds_final:
            preds, d_val, w_val = model(
                x_val,
                y=None,
                training=False,
                epoch=epoch
            )

            vloss = loss_fn(y_val_batch, preds).numpy()

            epoch_val_losses.append(vloss)
            all_y_epoch.extend(y_val_batch.numpy())
            all_probs_epoch.extend(preds.numpy().flatten())

            delta_batches.append(d_val.numpy())
            weight_batches.append(w_val.numpy())

        val_loss = float(np.mean(epoch_val_losses))
        val_auc = roc_auc_score(all_y_epoch, all_probs_epoch)

        val_loss_history.append(val_loss)
        val_auc_history.append(val_auc)

        final_deltas = np.mean(np.stack(delta_batches), axis=0)
        final_weights = np.mean(np.stack(weight_batches), axis=0)

        all_y = np.array(all_y_epoch)
        all_probs = np.array(all_probs_epoch)

        print(
            f"[{method_name} | Seed {seed}] "
            f"Epoch {epoch + 1}/{FULL_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

    y_pred = (all_probs > 0.5).astype(int)

    final_metrics = {
        "Method": method_name,
        "Seed": seed,
        "AUC": roc_auc_score(all_y, all_probs),
        "Accuracy": accuracy_score(all_y, y_pred),
        "Precision": precision_score(all_y, y_pred, zero_division=0),
        "Recall": recall_score(all_y, y_pred, zero_division=0),
        "F1": f1_score(all_y, y_pred, zero_division=0),
        "Attention_Discriminability": float(np.mean(np.abs(final_deltas))),
        "Weight_Entropy": float(
            -np.sum(final_weights * np.log(final_weights + 1e-9))
        )
    }

    print("\nFinal validation metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    model_save_path = os.path.join(
        OUTPUT_DIR,
        f"{method_name}_seed_{seed}_final_35epochs.keras"
    )

    model.save(model_save_path)

    np.save(
        os.path.join(OUTPUT_DIR, f"{method_name}_seed_{seed}_train_loss.npy"),
        np.array(train_loss_history)
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"{method_name}_seed_{seed}_val_loss.npy"),
        np.array(val_loss_history)
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"{method_name}_seed_{seed}_val_auc.npy"),
        np.array(val_auc_history)
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"{method_name}_seed_{seed}_final_deltas.npy"),
        np.array(final_deltas)
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"{method_name}_seed_{seed}_final_weights.npy"),
        np.array(final_weights)
    )

    print(f"\nSaved model: {model_save_path}")

    return final_metrics


# =========================================================
# RUN FINAL TRAINING FOR BOAUC AND AABO ACROSS TWO SEEDS
# =========================================================

def main():
    best_params_boauc = load_best_params(BEST_BOAUC_PATH)
    best_params_aabo = load_best_params(BEST_AABO_PATH)

    train_paths_raw, train_y_raw, val_paths, val_y = prepare_data()

    all_final_results = []

    for seed in SEEDS:
        boauc_metrics = run_final_training(
            best_params_boauc,
            method_name="WAF_X_Net_BOAUC",
            seed=seed,
            train_paths_raw=train_paths_raw,
            train_y_raw=train_y_raw,
            val_paths=val_paths,
            val_y=val_y
        )

        all_final_results.append(boauc_metrics)

        aabo_metrics = run_final_training(
            best_params_aabo,
            method_name="WAF_X_Net_AABO",
            seed=seed,
            train_paths_raw=train_paths_raw,
            train_y_raw=train_y_raw,
            val_paths=val_paths,
            val_y=val_y
        )

        all_final_results.append(aabo_metrics)

    # =====================================================
    # SAVE SEED-WISE RESULTS
    # =====================================================

    df_final = pd.DataFrame(all_final_results)

    seedwise_path = os.path.join(
        OUTPUT_DIR,
        "final_35epoch_validation_seedwise_results.csv"
    )

    df_final.to_csv(seedwise_path, index=False)

    metric_cols = [
        "AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Attention_Discriminability",
        "Weight_Entropy"
    ]

    # =====================================================
    # SAVE MEAN ± STD SUMMARY
    # =====================================================

    summary = (
        df_final
        .groupby("Method")[metric_cols]
        .agg(["mean", "std"])
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "final_35epoch_validation_mean_std_summary.csv"
    )

    summary.to_csv(summary_path)

    # =====================================================
    # SAVE FORMATTED SUMMARY TABLE
    # =====================================================

    rows = []

    for method in df_final["Method"].unique():
        sub = df_final[df_final["Method"] == method]
        row = {"Method": method}

        for metric in metric_cols:
            row[metric] = f"{sub[metric].mean():.4f} ± {sub[metric].std():.4f}"

        rows.append(row)

    df_table = pd.DataFrame(rows)

    table_path = os.path.join(
        OUTPUT_DIR,
        "final_35epoch_validation_table.csv"
    )

    df_table.to_csv(table_path, index=False)

    print("\nSeed-wise final validation results:")
    print(df_final)

    print("\nMean ± std validation summary:")
    print(summary)

    print("\nFormatted validation table:")
    print(df_table)

    print("\nSaved files:")
    print(seedwise_path)
    print(summary_path)
    print(table_path)


if __name__ == "__main__":
    main()
