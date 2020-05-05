import os
from third_party.mmaction.data_tools.climbing.crop_video_patches import *
class Tracklet(object):
    def __init__(self,id,box,action):
        self.id=id
        self.bbox=box
        self.action=action
        self.tracklet=[]
        self.frame_history=[]
    def update(self,bbox,frame_num):
        self.tracklet.append(bbox)
        self.frame_history.append(frame_num)
def generate_segments(root_path):
    label_path=os.path.join(root_path,'annotations','raw_labels')
    label_list=os.listdir(label_path)
    for label in label_list:
        id_pool=[]
        tracklets={}
        save_path = os.path.join(root_path, 'extracted_segments',os.path.splitext(label)[0])
        o=open(os.path.join(label_path,label),'r')
        label_lines=o.readlines()
        for line in label_lines:
            line=line.rstrip()
            line_list=line.split(',')
            frame_num=line_list[0]
            obj_nums=line_list[1]
            for instance in range(int(obj_nums)):
                obj_id=line_list[2+6*instance]
                bbox=line_list[3 + 6 * instance:7 + 6 * instance]
                action=line_list[7+6*instance]

                if obj_id+action not in id_pool:

                    id_pool.append(obj_id+action)
                    obj=Tracklet(obj_id,bbox,action)
                    tracklets[obj_id+action]=obj
                else:
                    tracklets[obj_id+action].update(bbox,frame_num)

        crop_patches(tracklets,save_path)
def crop_patches(tracklets,save_path):
    rawframes_dir='/home/rvlab/Documents/DRDvideo_processed/raw_frames/'
    for tracklet in tracklets.values():
        data=None
        action=tracklet.action
        if not os.path.isdir(os.path.join(save_path, action)):
            os.makedirs(os.path.join(save_path, action))
        target_id = tracklet.id
        base_video_name = os.path.split(save_path)[1]
        target_id, start_frame, end_frame = int(target_id), int(tracklet.frame_history[0]), int(tracklet.frame_history[-1])

        # data = np.loadtxt('/home/rvlab/Documents/hmdb_tracked/tracked_videos/walk/21_walk_h_cm_np1_fr_med_10.txt',delimiter=',')
        if len(tracklet.frame_history)>8:#threshold of min frames
            video_writer = cv2.VideoWriter(
                os.path.join(save_path, action,
                             '{}_{}_{}_{}.mp4'.format(base_video_name, target_id, start_frame, end_frame)),
                cv2.VideoWriter_fourcc(*'mp4v'), 14, (32, 64))
            for idx,frame in enumerate(tracklet.frame_history):
                bbox=list(map(int, tracklet.tracklet[idx]))
                row= [int(frame)]+[int(target_id)]+ bbox
                if data is None:
                    data=np.array(row)
                else:
                    data=np.vstack((data,np.array(row)))

            print('writing video {} target #{} action {}'.format(base_video_name, target_id ,action))
            # np.savetxt('{}_{}_before.txt'.format(video_name, target_id), data, fmt='%.2f')
            data = fill_gaps(data, target_id)
            # np.savetxt('{}_{}_after.txt'.format(video_name, target_id), data, fmt='%.2f')

            data = smooth(data, window_size=9)

            for detection in data:
                detection[3]=detection[3]-80#- cut_y
                box = detection[2:6]
                x_c, y_c = detection[2] + detection[4] / 2, detection[3] + detection[5] / 2
                w, h = detection[4], detection[5]
                img = cv2.imread(os.path.join(rawframes_dir, base_video_name+'_{:06d}.jpg'.format(int(detection[0]))))
                if img is None:
                    print('Frame {} Not Found'.format(
                        os.path.join(rawframes_dir, base_video_name + '_{:06d}.jpg'.format(int(detection[0])))))
                    break
                patch = crop(img, x_c, y_c, max(w, h))

                patch = cv2.resize(patch, (32, 64))
                video_writer.write(patch)

            video_writer.release()
if __name__ == '__main__':
    generate_segments('/home/rvlab/Documents/DRDvideo_processed/')