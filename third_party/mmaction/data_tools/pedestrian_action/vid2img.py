# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import threading

NUM_THREADS = 100
VIDEO_ROOT = '/home/rvlab/Documents/DRDvideo_processed/extracted_segments/'         # Downloaded webm videos
FRAME_ROOT = '/home/rvlab/Documents/DRDvideo_processed/processed_frames/'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video,ROOT_path, tmpl='%06d.jpg'):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(ROOT_path, video,
                                                                                             FRAME_ROOT, video[:-4])
    os.system(cmd)


def target(video_list,ROOT_path):
    for video in video_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-4]))
        extract(video,ROOT_path)


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)
    video_folder_list = os.listdir(VIDEO_ROOT)
    for video_folder in video_folder_list:
        actions_list=os.listdir(os.path.join(VIDEO_ROOT,video_folder))
        for action in actions_list:
            video_list=os.listdir(os.path.join(VIDEO_ROOT,video_folder,action))
            splits = list(split(video_list, NUM_THREADS))

            threads = []
            for i, sp in enumerate(splits):
                thread = threading.Thread(target=target, args=(sp,os.path.join(VIDEO_ROOT,video_folder,action)))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()