# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import threading

NUM_THREADS = 100
# VIDEO_ROOT = '/home/rvlab/Documents/DRDvideo_processed/extracted_segments/'         # Downloaded webm videos
# FRAME_ROOT = '/home/rvlab/Documents/DRDvideo_processed/processed_frames/'  # Directory for extracted frames
# VIDEO_ROOT = '/home/rvlab/Documents/DRDvideo_processed/video_segments/'         # Downloaded webm videos
# FRAME_ROOT = '/home/rvlab/Documents/DRDvideo_processed/processed/'
VIDEO_ROOT = '/home/rvlab/Documents/DRDvideo_processed/campus_extracted/'         # Downloaded webm videos
FRAME_ROOT = '/home/rvlab/Documents/DRDvideo_processed/campus_processed/'
def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video,ROOT_path, tmpl='%06d.jpg'):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=64:128 -q:v 0 \"{}/{}/%06d.jpg\"'.format(ROOT_path, video[0],
                                                                                             FRAME_ROOT,  video[1][:-4])
    os.system(cmd)


def target(video_list,ROOT_path):
    for video in video_list:
        num=os.path.split(ROOT_path)[1]
        action=os.path.split(os.path.split(ROOT_path)[0])[1]

        new_name=video[:-4]+'_'+action+'_'+num+'.mp4'
        os.makedirs(os.path.join(FRAME_ROOT,new_name[:-4]))
        extract([video,new_name],ROOT_path)


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT): #DRD DATA
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)
    actions_list=os.listdir(VIDEO_ROOT)
    for action in actions_list:
        num_list = os.listdir(os.path.join(VIDEO_ROOT,action))
        for num in num_list:
            video_list=os.listdir(os.path.join(VIDEO_ROOT,action,num))
            splits = list(split(video_list, NUM_THREADS))

            threads = []
            for i, sp in enumerate(splits):
                thread = threading.Thread(target=target, args=(sp,os.path.join(VIDEO_ROOT,action,num)))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()
    # if not os.path.exists(VIDEO_ROOT): #AVA DATA
    #     raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    # if not os.path.exists(FRAME_ROOT):
    #     os.makedirs(FRAME_ROOT)
    # type_list = os.listdir(VIDEO_ROOT)
    # for data_type in type_list:
    #     actions_list=os.listdir(os.path.join(VIDEO_ROOT,data_type))
    #
    #     for action in actions_list:
    #
    #         video_list=os.listdir(os.path.join(VIDEO_ROOT,data_type,action))
    #         splits = list(split(video_list, NUM_THREADS))
    #
    #         threads = []
    #         for i, sp in enumerate(splits):
    #             thread = threading.Thread(target=target, args=(sp,os.path.join(VIDEO_ROOT,data_type,action)))
    #             thread.start()
    #             threads.append(thread)
    #
    #         for thread in threads:
    #             thread.join()