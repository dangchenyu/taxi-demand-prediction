import os
import cv2
import numpy as np

def get_video_writer(save_video_path, width, height):
    if save_video_path != '':
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(width), int(height)))
    else:
        class MuteVideoWriter():
            def write(self, *args, **kwargs):
                pass

            def release(self):
                pass

        return MuteVideoWriter()


if __name__ == '__main__':
    base_path = '/home/rvlab/Documents/DRDvideo_processed/campus_processed/'
    list_path = os.listdir(base_path)
    id_count = 0
    prob_dict={'stand':0,'stand-look':0.7,'weak-stand-look':0.3,'wave':1,'walk':0}
    cap = cv2.VideoCapture("/home/rvlab/Documents/DRDvideos/campus/campus_data.mp4")
    # with open(os.path.join('/home/rvlab/Documents/DRDvideos/campus/track_result', 'all.txt'), 'r') as o:
    lines=np.loadtxt(os.path.join('/home/rvlab/Documents/DRDvideos/campus/track_result', 'all.txt'),delimiter=',')
    # lines = o.readlines()
    for val in ['1-4','5-8','9-12','13-16','17-19']:
        id_pool = {}
        id_last = -1
        id_count = 0

        video_writer = get_video_writer(
            os.path.join('/home/rvlab/Documents/DRDvideos/campus/', 'gt' + val + '.mp4'), 1280, 720)

        with open(os.path.join('/home/rvlab/Documents/DRDvideo_processed/annotations/', 'gt_anno' + val + '.txt'),
                  'w+') as b:
            for folder in list_path:
                if val in folder:
                    folder_list = folder.split('_')
                    id = float(folder_list[2])
                    id_pool[id] = folder_list[5]
            id_list=sorted(list(id_pool.keys()))
            # id_list=.sort()
            for exist_id in id_list:
                id_lines = lines[np.where(lines[:, 1] == exist_id)]
                id_lines = id_lines[id_lines[:, 0].argsort()].tolist()
                for line in id_lines:
                    id = line[1]
                    bbox=line[2:6]
                    if id != id_last:

                        print("writing", val, id_count, id_pool[id])
                        b.write('{},{},{},{}\n'.format(id_count, id_pool[id],prob_dict[id_pool[id]],bbox))
                        id_count += 1
                        id_last = id
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(line[0]))

                    a, img = cap.read()
                    video_writer.write(img)
        b.close()
        video_writer.release()
    # o.close()
