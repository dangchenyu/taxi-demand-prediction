#! /usr/bin/bash env

python extract_original_frames.py --annotation_file ../../data/climbing/annotations/climb.txt --videos_dir ../../data/climbing/original_videos --output_dir ../../data/climbing/original_frames
python extract_original_frames.py --annotation_file ../../data/climbing/annotations/notclimb.txt --videos_dir ../../data/climbing/original_videos --output_dir ../../data/climbing/original_frames