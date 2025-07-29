import cv2
import os
import glob
import argparse
import random

def has_adjacent_dupes(seq):
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Create video from images with duplicated frames (no adjacent duplicates)")
    parser.add_argument("--image_folder", type=str, default="video_lp_images",
                        help="Folder path containing images (default: current folder)")
    parser.add_argument("--output_video", type=str, default="videos/test_demo.mp4",
                        help="Output video filename (default: test_parking_simulation.mp4)")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second of the output video (default: 1)")

    args = parser.parse_args()

    image_extensions = ["jpg", "jpeg", "png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, f"*.{ext}")))
    image_paths = list(set(image_paths))
    if not image_paths:
        raise ValueError(f"No images found in directory {args.image_folder}")

    video_sequence = image_paths + image_paths  # Two of each
    attempts = 0
    max_attempts = 1000

    if len(set(image_paths)) <= 2:
        # Too few unique images: accept inevitable adjacent duplicate(s)
        print("Too few unique images to guarantee no adjacent duplicates. Proceeding anyway.")
        random.shuffle(video_sequence)
    else:
        while attempts < max_attempts:
            random.shuffle(video_sequence)
            if not has_adjacent_dupes(video_sequence):
                break
            attempts += 1
        else:
            print("Warning: Could not create a shuffle without adjacent duplicates after many tries. Proceeding anyway.")

    img = cv2.imread(video_sequence[0])
    if img is None:
        raise ValueError(f"First image {video_sequence[0]} could not be read (check format)!")
    height, width, layers = img.shape
    os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))

    for img_path in video_sequence:
        img = cv2.imread(img_path)
        if img is not None:
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height))
            video.write(img)
        else:
            print(f"Warning: Unable to read image {img_path}, skipping.")

    video.release()
    print(f'Video created: {args.output_video}')

if __name__ == "__main__":
    main()

