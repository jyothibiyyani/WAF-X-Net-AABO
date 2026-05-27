# Dataset Preparation

This repository does not redistribute Celeb-DF v2 or FaceForensics++.

Users must obtain the datasets from their official sources and comply with the corresponding dataset licenses. This repository provides dataset usage information, expected folder structure, frame-sampling details, preprocessing steps, video-level split files, and subset identifiers required to reproduce the experiments reported in the manuscript.

---

## 1. Dataset Usage Overview

The experiments use two publicly available datasets:

1. Celeb-DF v2
2. FaceForensics++ Face2Face

Celeb-DF v2 is used for:

- hyperparameter optimization,
- full 35-epoch retraining,
- validation-stage comparison,
- and strictly held-out testing.

FaceForensics++ Face2Face is used only for:

- cross-dataset zero-shot evaluation,
- and 10% target-domain fine-tuning evaluation.

The datasets themselves are not uploaded to this repository.

---

## 2. Celeb-DF v2 Folder Structure

After frame extraction, Celeb-DF v2 frames should be organized as follows:

```text
celeb_df frames/
├── celeb_df_V2_real_frames/
│   ├── id0_0000/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   ├── id1_0000/
│   │   ├── 00000.jpg
│   │   └── ...
│   └── ...
└── celeb_df_V2_fake_frames/
    ├── id0_id1_0000/
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    ├── id1_id2_0000/
    │   ├── 00000.jpg
    │   └── ...
    └── ...
```

Example real frame path:

```text
celeb_df_V2_real_frames/id0_0000/00000.jpg
```

Corresponding video-level path used in split CSV files:

```text
celeb_df_V2_real_frames/id0_0000
```

Example fake frame path:

```text
celeb_df_V2_fake_frames/id0_id1_0000/00000.jpg
```

Corresponding video-level path used in split CSV files:

```text
celeb_df_V2_fake_frames/id0_id1_0000
```

The split files contain video-folder paths, not individual frame paths.

---

## 3. Celeb-DF v2 Experimental Protocol

The Celeb-DF v2 experimental protocol used in the manuscript is:

```text
Train/validation pool: 400 real + 400 fake videos
Training split: 320 real + 320 fake videos
Validation split: 80 real + 80 fake videos
Held-out test set: 100 real + 100 fake videos
Frames sampled per video: 30
Held-out frame-sampling seeds: 42, 123
```

The 400 real + 400 fake videos are used only for optimization and full retraining. This pool is split using a fixed 80:20 video-level train-validation split.

The separate held-out Celeb-DF v2 test set contains 100 real + 100 fake videos. This held-out test set is never used during:

- hyperparameter optimization,
- validation-stage comparison,
- final model selection,
- or training.

All Celeb-DF v2 splits are performed at the video level before frame sampling.

---

## 4. Celeb-DF v2 Frame Counts

For the train-validation pool:

```text
Training videos: 320 real + 320 fake = 640 videos
Validation videos: 80 real + 80 fake = 160 videos
Frames per video: 30
```

Approximate frame counts:

```text
Training frames: 640 × 30 = 19,200 frames
Validation frames: 160 × 30 = 4,800 frames
```

For held-out testing:

```text
Held-out videos: 100 real + 100 fake = 200 videos
Frames per video: 30
Nominal held-out frames: 200 × 30 = 6,000 frames
```

After excluding corrupted or invalid frames during preprocessing, the held-out evaluation contains:

```text
5,998 valid frames
```

---

## 5. FaceForensics++ Face2Face Folder Structure

After frame extraction, FaceForensics++ Face2Face frames should be organized as follows:

```text
FF++_frames/
├── Original_sequences/
│   ├── 000/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ...
│   ├── 001/
│   │   ├── frame_0000.jpg
│   │   └── ...
│   └── ...
└── Face2Face/
    ├── 000_003/
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    ├── 001_004/
    │   ├── 00000.jpg
    │   └── ...
    └── ...
```

Example real frame path:

```text
Original_sequences/000/frame_0000.jpg
```

Corresponding video-level path used in split CSV files:

```text
Original_sequences/000
```

Example fake frame path:

```text
Face2Face/000_003/00000.jpg
```

Corresponding video-level path used in split CSV files:

```text
Face2Face/000_003
```

The split files contain video-folder paths, not individual frame paths.

---

## 6. FaceForensics++ Face2Face Experimental Protocol

The manuscript uses a selected FaceForensics++ Face2Face experimental subset, not the complete FaceForensics++ dataset.

The selected subset contains:

```text
500 real videos + 500 fake videos
```

This reduced Face2Face subset is used for computationally controlled cross-dataset evaluation.

For each cross-dataset seed:

```text
Fine-tuning videos: 50 real + 50 fake videos
Evaluation videos: 450 real + 450 fake videos
Frames sampled per video: 30
```

The five cross-dataset target-domain split and frame-sampling seeds are:

```text
7, 42, 123, 2024, 2025
```

The FaceForensics++ experiment includes two settings:

1. Zero-shot evaluation:
   - the Celeb-DF v2 trained model is evaluated on FaceForensics++ Face2Face without target-domain fine-tuning.

2. 10% video-level fine-tuning:
   - 50 real + 50 fake FaceForensics++ Face2Face videos are used for fine-tuning,
   - 450 real + 450 fake videos are reserved for evaluation.

All FaceForensics++ splits are performed at the video level before frame sampling.

---

## 7. FaceForensics++ Frame Counts

For each cross-dataset seed:

```text
Fine-tuning videos: 50 real + 50 fake = 100 videos
Evaluation videos: 450 real + 450 fake = 900 videos
Frames per video: 30
```

Approximate frame counts:

```text
Fine-tuning frames: 100 × 30 = 3,000 frames
Evaluation frames: 900 × 30 = 27,000 frames
```

---

## 8. Frame Sampling Protocol

For all experiments:

1. Videos are split at the video level.
2. Frame sampling is performed only after video-level partitioning.
3. Up to 30 frames are sampled from each selected video.
4. The same video never appears in more than one split.
5. Invalid or corrupted frames are skipped during preprocessing.

The frame-sampling code is provided in:

```text
preprocessing/frame_sampling.py
```

The model input size is:

```text
224 × 224 × 3
```

---

## 9. Image Preprocessing

Each sampled frame is processed as follows:

```python
img = tf.io.read_file(path)
img = tf.io.decode_image(img, channels=3, expand_animations=False)
img = tf.image.resize(img, (224, 224))
img = tf.keras.applications.xception.preprocess_input(img)
```

This preprocessing is used consistently for:

- training,
- validation,
- held-out testing,
- cross-dataset evaluation,
- and ablation experiments.

---

## 10. Local Path Mapping

The split CSV files use relative video-folder paths instead of absolute system paths.

Example local roots:

```text
CELEBDF_ROOT = C:/Users/nagir/OneDrive/Documents/frames/celeb_df frames
FFPP_ROOT = C:/Users/nagir/OneDrive/Documents/frames/FF++_frames
```

The scripts should join the dataset root with the relative path stored in each split file.

Example:

```python
video_path = os.path.join(CELEBDF_ROOT, relative_path)
```

Do not upload raw videos, extracted frames, or absolute local paths such as:

```text
C:/Users/nagir/OneDrive/Documents/...
```

Only relative video-folder paths should be included in GitHub split files.

---

## 11. Required Split Files

The following exact video-level split files are provided in the repository:

```text
splits/
├── celebdf_train_val_split.csv
├── celebdf_heldout_100real_100fake.csv
├── ffpp_face2face_subset_500real_500fake.csv
└── ffpp_face2face_5seed_splits/
    ├── seed_7_split.csv
    ├── seed_42_split.csv
    ├── seed_123_split.csv
    ├── seed_2024_split.csv
    └── seed_2025_split.csv
```

These files should contain video-folder-level paths, not frame-level paths.

Correct examples:

```text
celeb_df_V2_fake_frames/id0_id1_0000
celeb_df_V2_real_frames/id0_0000
Original_sequences/000
Face2Face/000_003
```

Incorrect examples:

```text
celeb_df_V2_fake_frames/id0_id1_0000/00000.jpg
celeb_df_V2_real_frames/id0_0000/00000.jpg
Original_sequences/000/frame_0000.jpg
Face2Face/000_003/00000.jpg
```

---

## 12. Split CSV Format

Each split CSV uses the following columns:

```text
dataset,video_id,relative_path,label,split,seed
```

Column meanings:

- `dataset`: Dataset name.
- `video_id`: Video folder identifier.
- `relative_path`: Path relative to dataset root.
- `label`: 0 for real and 1 for fake.
- `split`: train, validation, heldout, selected_subset, finetune, or evaluation.
- `seed`: Random seed used for the split; use `NA` for fixed non-seed splits.

---

## 13. Split Verification

A helper script is provided to verify split counts:

```text
tools/check_splits.py
```

Run:

```bash
python tools/check_splits.py
```

This script checks whether the split files contain the expected columns and prints counts by split and label.

---

## 14. Important Notes

- Do not redistribute Celeb-DF v2 or FaceForensics++ videos.
- Do not upload extracted frames.
- Do not upload absolute local system paths.
- Use relative video-folder paths in split CSV files.
- Split videos before sampling frames.
- Keep held-out videos separate from training and validation videos.
- Keep cross-dataset fine-tuning and evaluation videos separate within each seed.
