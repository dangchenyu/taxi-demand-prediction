#! /usr/bin/bash env

python crop_video_patches.py --annotation_file ../../data/climbing/annotations/climb.txt --rawframes_dir ../../data/climbing/original_frames --track_dir ../../data/climbing/tracks --output_dir ../../data/climbing/videos/climb
python crop_video_patches.py --annotation_file ../../data/climbing/annotations/notclimb.txt --rawframes_dir ../../data/climbing/original_frames --track_dir ../../data/climbing/tracks --output_dir ../../data/climbing/videos/notclimb