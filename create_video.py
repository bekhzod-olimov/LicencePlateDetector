import cv2
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create video from images with duplicated frames")
    parser.add_argument("--image_folder", type=str, default=".",
                        help="Folder path containing images (default: current folder)")
    parser.add_argument("--output_video", type=str, default="test_parking_simulation.mp4",
                        help="Output video filename (default: test_parking_simulation.mp4)")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second of the output video (default: 1)")

    args = parser.parse_args()

    # Supported image extensions
    image_extensions = ["jpg", "jpeg", "png"]

    # Collect all images (case-insensitive)
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, f"*.{ext}")))

    # Sort paths so duplicates pair up, then duplicate the sequence
    image_paths.sort()
    double_image_paths = image_paths + image_paths  # Simulate 'IN', then 'OUT' for each

    # Sanity check
    if not double_image_paths:
        raise ValueError(f"No images found in directory {args.image_folder}")

    # Read the size from the first image
    img = cv2.imread(double_image_paths[0])
    if img is None:
        raise ValueError(f"First image {double_image_paths[0]} could not be read (check format)!")
    height, width, layers = img.shape

    # Initialize video writer    
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))

    # Write each image to the video
    for img_path in double_image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            # Optional: resize if dimension mismatch
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height))
            video.write(img)
        else:
            print(f"Warning: Unable to read image {img_path}, skipping.")

    video.release()
    print(f'Video created: {args.output_video}')


if __name__ == "__main__":
    main()
