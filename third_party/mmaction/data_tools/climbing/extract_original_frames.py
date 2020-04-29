import os
import cv2
import tqdm
import logging
import argparse


def range_overlap_adjust(list_ranges):
    overlap_corrected = []
    for start, stop in sorted(list_ranges):
        if overlap_corrected and start - 1 <= overlap_corrected[-1][1] and stop >= overlap_corrected[-1][1]:
            overlap_corrected[-1] = min(overlap_corrected[-1][0], start), stop
        elif overlap_corrected and start <= overlap_corrected[-1][1] and stop <= overlap_corrected[-1][1]:
            continue
        else:
            overlap_corrected.append((start, stop))
    return overlap_corrected


def extract_rawframes(video_path, frame_ranges, output_dir):
    logging.info('')
    logging.info('Extracting frames from {}'.format(video_path))
    extention = video_path.split('.')[-1]
    video_filename = os.path.splitext(video_path.split('/')[-1])[0]
    if 'wave' in video_filename:
        action = 'wave'
    else:
        action = 'walk'
    rawframes_dir = os.path.join(output_dir, action, video_filename)
    if not os.path.isdir(rawframes_dir):
        os.makedirs(rawframes_dir)
    capture = cv2.VideoCapture(video_path)
    n = 0
    range_pointer = 0
    start_milestones = [frame_range[0] for frame_range in frame_ranges]
    end_milestones = [frame_range[1] for frame_range in frame_ranges]
    write = False
    with tqdm.tqdm(total=cv2.CAP_PROP_FRAME_COUNT) as pbar:
        while True:
            n += 1
            if range_pointer == len(start_milestones):
                logging.info('No more targets to crop')
                break
            if n == start_milestones[range_pointer]:
                logging.info('Starting writing at frame #{}'.format(start_milestones[range_pointer]))
                write = True
            if n == end_milestones[range_pointer] + 1:
                logging.info('Stopping writing at frame #{}'.format(end_milestones[range_pointer]))
                write = False
                range_pointer += 1
            pbar.update(1)
            ret, frame = capture.read()
            if not ret:
                logging.info('Capture ended')
                break
            if write:
                cv2.imwrite('{}/{}.png'.format(rawframes_dir, n), frame)
    logging.info('Extracted {} raw frames from video file {}'.format(n, video_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raw frame extractor')
    parser.add_argument('--annotation_file', type=str, help='Path to the annotation file')
    parser.add_argument('--output_dir', type=str, help='Path to output original frames')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    annotation_file = open(args.annotation_file, 'r')

    frame_ranges = []
    for index, line in enumerate(annotation_file):
        video_name, target_id, start_frame, end_frame = line.strip().split(',')
        target_id, start_frame, end_frame = int(target_id), int(start_frame), int(end_frame)
        extract_rawframes(video_name, frame_ranges, args.output_dir)

        frame_ranges = [(start_frame, end_frame)]

        print(line[0])
    # Extract last video
