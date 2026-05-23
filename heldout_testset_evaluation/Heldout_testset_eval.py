# =========================================================
# HELD-OUT CELEB-DF v2 EVALUATION
# =========================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

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

FRAME_SAMPLING_SEEDS = [42, 123]

# Saved AABO model from final 35-epoch training.
MODEL_PATH = "final_35epoch_results/WAF_X_Net_AABO_seed_42_final_35epochs.keras"

# CSV file containing strictly held-out video folders.
# Required columns:
#     video_dir,label
#
# label:
#     0 = real
#     1 = fake
#
# Example:
#     /path/to/heldout_real_video_001,0
#     /path/to/heldout_fake_video_001,1

HELDOUT_CSV = "splits/celebdf_heldout_100real_100fake.csv"

OUTPUT_DIR = "heldout_evaluation_results"
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


def load_heldout_split(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Provide the held-out split CSV file."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"video_dir", "label"}
    missing = required_cols.difference(df.columns)

    if missing:
        raise KeyError(f"Missing required columns in {csv_path}: {missing}")

    video_dirs = df["video_dir"].tolist()
    labels = df["label"].tolist()

    labels_clean = []

    for y in labels:
        if isinstance(y, str):
            y_lower = y.strip().lower()

            if y_lower in ["real", "0"]:
                labels_clean.append(0)
            elif y_lower in ["fake", "1"]:
                labels_clean.append(1)
            else:
                raise ValueError(f"Invalid label value: {y}")
        else:
            labels_clean.append(int(y))

    for video_dir in video_dirs:
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Video folder not found: {video_dir}")

    return video_dirs, labels_clean


def sample_frames(video_dirs, labels, max_frames=30, seed=42):
    rng = random.Random(seed)

    paths = []
    y = []
    video_ids = []

    for vid, label in zip(video_dirs, labels):
        frames = sorted(os.listdir(vid))
        frames = [
            f for f in frames
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(frames) == 0:
            continue

        if len(frames) > max_frames:
            selected_frames = rng.sample(frames, max_frames)
            selected_frames = sorted(selected_frames)
        else:
            selected_frames = frames

        for f in selected_frames:
            full = os.path.join(vid, f)

            if is_valid_jpeg(full):
                paths.append(full)
                y.append(label)
                video_ids.append(os.path.basename(vid))

    return paths, y, video_ids


def load_and_preprocess_eval(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.xception.preprocess_input(img)

    return img, tf.cast(label, tf.float32)


def make_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


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
# LOAD MODEL
# =========================================================

def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    custom_objects = {
        "XceptionDWAF": XceptionDWAF
    }

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )
    except TypeError:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )

    return model


# =========================================================
# EVALUATION
# =========================================================

def evaluate_heldout(model, seed):
    set_seed(seed)

    print("\n====================================")
    print(f"Held-out evaluation | Frame seed {seed}")
    print("====================================")

    video_dirs, labels = load_heldout_split(HELDOUT_CSV)

    paths, y_true, video_ids = sample_frames(
        video_dirs,
        labels,
        max_frames=MAX_FRAMES,
        seed=seed
    )

    print(f"Held-out videos: {len(video_dirs)}")
    print(f"Valid held-out frames: {len(paths)}")
    print(f"Real frames: {sum(np.array(y_true) == 0)}")
    print(f"Fake frames: {sum(np.array(y_true) == 1)}")

    ds = make_dataset(paths, y_true)

    all_probs = []
    all_y = []
    all_paths = []
    all_video_ids = []

    delta_batches = []
    weight_batches = []

    start_idx = 0

    for x_batch, y_batch in ds:
        preds, deltas, weights = model(
            x_batch,
            y=None,
            training=False,
            epoch=FULL_EPOCHS
        )

        batch_probs = preds.numpy().flatten()

        batch_size = len(batch_probs)
        batch_paths = paths[start_idx:start_idx + batch_size]
        batch_video_ids = video_ids[start_idx:start_idx + batch_size]
        start_idx += batch_size

        all_probs.extend(batch_probs)
        all_y.extend(y_batch.numpy())
        all_paths.extend(batch_paths)
        all_video_ids.extend(batch_video_ids)

        delta_batches.append(deltas.numpy())
        weight_batches.append(weights.numpy())

    all_probs = np.array(all_probs)
    all_y = np.array(all_y).astype(int)

    y_pred = (all_probs > 0.5).astype(int)

    final_deltas = np.mean(np.stack(delta_batches), axis=0)
    final_weights = np.mean(np.stack(weight_batches), axis=0)

    metrics = {
        "Seed": seed,
        "Frames": len(all_y),
        "AUC": roc_auc_score(all_y, all_probs),
        "Accuracy": accuracy_score(all_y, y_pred),
        "Precision_macro": precision_score(
            all_y,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall_macro": recall_score(
            all_y,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "F1_macro": f1_score(
            all_y,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "Attention_Discriminability": float(np.mean(np.abs(final_deltas))),
        "Weight_Entropy": float(
            -np.sum(final_weights * np.log(final_weights + 1e-9))
        )
    }

    print("\nHeld-out metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    predictions_df = pd.DataFrame({
        "frame_path": all_paths,
        "video_id": all_video_ids,
        "label": all_y,
        "prob_fake": all_probs,
        "pred": y_pred
    })

    pred_path = os.path.join(
        OUTPUT_DIR,
        f"heldout_predictions_seed_{seed}.csv"
    )

    predictions_df.to_csv(pred_path, index=False)

    np.save(
        os.path.join(OUTPUT_DIR, f"heldout_deltas_seed_{seed}.npy"),
        np.array(final_deltas)
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"heldout_weights_seed_{seed}.npy"),
        np.array(final_weights)
    )

    print(f"Saved predictions: {pred_path}")

    return metrics


# =========================================================
# RUN HELD-OUT EVALUATION ACROSS FRAME-SAMPLING SEEDS
# =========================================================

def main():
    model = load_trained_model(MODEL_PATH)

    all_results = []

    for seed in FRAME_SAMPLING_SEEDS:
        result = evaluate_heldout(model, seed)
        all_results.append(result)

    df_results = pd.DataFrame(all_results)

    seedwise_path = os.path.join(
        OUTPUT_DIR,
        "heldout_seedwise_results.csv"
    )

    df_results.to_csv(seedwise_path, index=False)

    metric_cols = [
        "AUC",
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "Attention_Discriminability",
        "Weight_Entropy"
    ]

    summary = df_results[metric_cols].agg(["mean", "std"]).T

    summary_path = os.path.join(
        OUTPUT_DIR,
        "heldout_mean_std_summary.csv"
    )

    summary.to_csv(summary_path)

    rows = []

    row = {"Evaluation": "Held-out Celeb-DF v2"}

    for metric in metric_cols:
        row[metric] = (
            f"{df_results[metric].mean():.4f} ± "
            f"{df_results[metric].std():.4f}"
        )

    rows.append(row)

    df_table = pd.DataFrame(rows)

    table_path = os.path.join(
        OUTPUT_DIR,
        "heldout_evaluation_table.csv"
    )

    df_table.to_csv(table_path, index=False)

    print("\nSeed-wise held-out results:")
    print(df_results)

    print("\nMean ± std held-out summary:")
    print(summary)

    print("\nFormatted held-out table:")
    print(df_table)

    print("\nSaved files:")
    print(seedwise_path)
    print(summary_path)
    print(table_path)


if __name__ == "__main__":
    main()
