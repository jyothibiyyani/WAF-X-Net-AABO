# =========================================================
# preprocessing/extract_frames.py
# Generic video-to-frame extraction utility for WAF-X-Net
# =========================================================

import os
import cv2
import argparse
import pandas as pd


def extract_frames_from_videos(
    videos_dir,
    output_dir,
    resize_width=224,
    resize_height=224,
    video_extensions=(".mp4", ".avi", ".mov", ".mkv"),
    save_every_n_frames=1
):
    """
    Extract frames from videos and save them into one folder per video.

    Parameters
    ----------
    videos_dir : str
        Directory containing raw videos.
    output_dir : str
        Directory where extracted frames will be saved.
    resize_width : int
        Output frame width.
    resize_height : int
        Output frame height.
    video_extensions : tuple
        Accepted video file extensions.
    save_every_n_frames : int
        Save every nth frame. Use 1 to save all frames.

    Notes
    -----
    The manuscript experiments use 224 x 224 input resolution and sample
    30 frames per video after video-level partitioning. This script only
    extracts and resizes frames. The 30-frame sampling is handled separately
    in `preprocessing/frame_sampling.py`.
    """

    os.makedirs(output_dir, exist_ok=True)

    extraction_log = []

    video_files = [
        f for f in sorted(os.listdir(videos_dir))
        if f.lower().endswith(video_extensions)
    ]

    if len(video_files) == 0:
        raise FileNotFoundError(f"No video files found in: {videos_dir}")

    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        print(f"Processing video: {video_file}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            extraction_log.append({
                "video_file": video_file,
                "status": "failed_open",
                "saved_frames": 0
            })
            continue

        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        frame_index = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % save_every_n_frames == 0:
                resized_frame = cv2.resize(
                    frame,
                    (resize_width, resize_height),
                    interpolation=cv2.INTER_AREA
                )

                frame_output_path = os.path.join(
                    video_output_dir,
                    f"frame_{saved_count:04d}.jpg"
                )

                success = cv2.imwrite(frame_output_path, resized_frame)

                if success:
                    saved_count += 1

            frame_index += 1

        cap.release()

        print(f"Saved {saved_count} frames from {video_file}")

        extraction_log.append({
            "video_file": video_file,
            "video_folder": video_name,
            "status": "success",
            "total_read_frames": frame_index,
            "saved_frames": saved_count
        })

    log_path = os.path.join(output_dir, "frame_extraction_log.csv")
    pd.DataFrame(extraction_log).to_csv(log_path, index=False)

    print(f"\nFrame extraction completed.")
    print(f"Extraction log saved to: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos for WAF-X-Net experiments."
    )

    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Directory containing input videos."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where extracted frames will be saved."
    )

    parser.add_argument(
        "--resize_width",
        type=int,
        default=224,
        help="Frame resize width. Default: 224."
    )

    parser.add_argument(
        "--resize_height",
        type=int,
        default=224,
        help="Frame resize height. Default: 224."
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=1,
        help="Save every nth frame. Default: 1."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extract_frames_from_videos(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        save_every_n_frames=args.save_every_n_frames
    )
