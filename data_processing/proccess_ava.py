import csv
import os
from data_processing.generate_tracklet import Tracklet
import cv2
import numpy as np
from third_party.mmaction.data_tools.pedestrian_action.crop_video_patches import fill_gaps, smooth

csv_path = '/media/rvlab/B8C40EFCC40EBC9E/ava/ava_v2.2'


class Tracklet_ava(Tracklet):

    def update(self, bbox, frame_num, action):
        self.tracklet.append(bbox)
        self.frame_history.append(frame_num)
        self.action.append(action)

    def pop(self):
        self.tracklet.pop(-1)
        self.frame_history.pop(-1)
        self.action.pop(-1)


def load_csv(csv_path):
    last_name = ''

    train_path = os.path.join(csv_path, 'ava_train_v2.2.csv')
    val_path = os.path.join(csv_path, 'ava_val_v2.2.csv')
    action_dict = {'12': 'standing', '14': 'walking', '69': 'waving'}
    video_dict = {}
    with open(val_path, 'r') as o:
        reader = csv.reader(o)
        # np_array_dict = {}
        # for line in reader:
        #     video_name = line[0]
        #     if video_name != last_name:
        #         print('processing',video_name)
        #         if last_name!='':
        #             np_array_dict[last_name]=np_array
        #         last_name = video_name
        #         np_array = np.array([int(line[6]),int(line[-1])])
        #     else:
        #         action_np = int(line[6])
        #         person_np = int(line[-1])
        #         np_array = np.vstack((np_array, np.array([action_np, person_np])))
        for line in reader:
            video_name = line[0]
            if video_name != last_name:

                cap = cv2.VideoCapture('/media/rvlab/B8C40EFCC40EBC9E/videos_trimmed_trainval/' + video_name + '.mp4')
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if last_name != '':
                    video_dict[last_name] = id_dict
                last_name = video_name
                id_dict = {}


            else:
                last_name = video_name

            frame_num = (int(line[1]) - 840) * fps

            xyxy = [float(line[2]) * frame_width, float(line[3]) * frame_height, float(line[4]) * frame_width,
                    float(line[5]) * frame_height]

            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            xywh = [float(line[2]) * frame_width, float(line[3]) * frame_height, w, h]
            # cv2.rectangle(b, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 1)
            # cv2.imshow('test', b)
            # cv2.waitKey(0)
            action_type = line[6]
            person_id = line[7]
            if person_id not in id_dict.keys():
                if h>=2.5*w:
                    if action_type == '12':
                        action_type = 'standing'
                    elif action_type == '14':
                        action_type = 'walking'
                    elif action_type == '69':
                        action_type = 'waving'
                    obj = Tracklet_ava(person_id, xywh, frame_num, action_type)
                    id_dict[person_id] = obj




            else:
                if action_type == '12':
                    action_type = 'standing'
                elif action_type == '14':
                    action_type = 'walking'
                elif action_type == '69':
                    action_type = 'waving'
                if id_dict[person_id].action[-1] == 'waving':
                    if action_type != 'waving':
                        continue
                if action_type == 'waving':
                    if id_dict[person_id].action[-1] != 'waving':
                        id_dict[person_id].pop()
                id_dict[person_id].update(xywh, frame_num, action_type)


        o.close()
    print('dict generated')
    return video_dict


def crop_imgs(img, x_c, y_c, window_size):
    x_base = 0
    y_base = 0
    padded_img = img
    half_w = int(window_size[0] // 2)
    half_h = int(window_size[1] // 2)
    if x_c < half_w:
        padded_img = cv2.copyMakeBorder(padded_img, 0, 0, half_w, 0, borderType=cv2.BORDER_REFLECT)
        x_base = half_w
    if x_c > img.shape[1] - half_w:
        padded_img = cv2.copyMakeBorder(padded_img, 0, 0, 0, half_w, borderType=cv2.BORDER_REFLECT)
    if y_c < half_h:
        padded_img = cv2.copyMakeBorder(padded_img, half_h, 0, 0, 0, borderType=cv2.BORDER_REFLECT)
        y_base = half_h
    if y_c > img.shape[0] - half_h:
        padded_img = cv2.copyMakeBorder(padded_img, 0, half_h, 0, 0, borderType=cv2.BORDER_REFLECT)
    return padded_img[int(y_base + y_c - half_h): int(y_base + y_c + half_h),
           int(x_base + x_c - half_w): int(x_base + x_c + half_w), :]


def filter_video_dict(video_dict, save_path, frame_path):
    count_walk = 0
    count_stand = 0
    count_wave = 0
    for video in video_dict.keys():
        base_video_name = video
        for obj in video_dict[video].keys():
            target_id = obj
            action_list = video_dict[video][obj].action

            if len(set(action_list))==1:
                if action_list[0]=='standing' or action_list[0]=='walking' or action_list[0]=='waving':
                    if action_list[0]=='standing':
                        count_stand += 1
                        if count_stand%5==0:
                            write_video(action_list,video,obj,save_path,target_id,base_video_name,frame_path)
                    if action_list[0]=='walking':
                        count_walk += 1
                        if count_walk%10==0:
                            write_video(action_list,video,obj,save_path,target_id,base_video_name,frame_path)
                    if action_list[0]=='waving':
                        count_wave += 1
                        write_video(action_list, video, obj, save_path, target_id, base_video_name, frame_path)



            else:
                continue
    print(count_stand, count_walk, count_wave)

def write_video(action_list,video,obj,save_path,target_id,base_video_name,frame_path):
    bbox = []
    last_action = ''
    if_first = True
    len_action = len(action_list)
    for index, action in enumerate(action_list):
        if if_first:
            start_frame = int(video_dict[video][obj].frame_history[index] - 30)
            bbox.append(video_dict[video][obj].tracklet[index])
        if index + 1 < len_action:
            if action == action_list[index + 1]:
                if (video_dict[video][obj].frame_history[index] + 30) == \
                        video_dict[video][obj].frame_history[
                            index + 1]:
                    bbox.append(video_dict[video][obj].tracklet[index + 1])
                    end_frame = int(
                        video_dict[video][obj].frame_history[index + 1] + 30)
                    if_first = False
                    continue
        if not os.path.isdir(os.path.join(save_path, action)):
            os.makedirs(os.path.join(save_path, action))
        if if_first:
            start_frame, end_frame = int(video_dict[video][obj].frame_history[index] - 30), int(
                video_dict[video][obj].frame_history[index] + 30)
        if_first = True
        video_writer = cv2.VideoWriter(
            os.path.join(save_path, action,
                         '{}_{}_{}_{}_{}.mp4'.format(video, target_id, action,
                                                     start_frame, end_frame)),
            cv2.VideoWriter_fourcc(*'mp4v'), 15, (32, 64))
        box_2_frames = []
        total_frames = end_frame - start_frame
        num_boxs = len(bbox)
        for box_ind, box_item in enumerate(bbox):
            if box_ind == 0:
                box_2_frames = [bbox[0]] * 30
                continue
            gap = np.array(box_item) - np.array(bbox[box_ind - 1])
            every_frame_gap = gap / 30
            box_2_frames += [(box_2_frames[-1] + every_frame_gap * frame_id).tolist() for frame_id in
                             range(1, 31)]
        if num_boxs > 1:
            print(base_video_name, obj, action)
        data = None
        i = 1
        frame_num = 0
        for frame in range(start_frame, end_frame + 1, 2):

            if frame <= start_frame + 30 * i:
                try:
                    row = [int(frame)] + [int(target_id)] + list(map(int, box_2_frames[frame_num]))
                    frame_num += 2
                except:
                    row = [int(frame)] + [int(target_id)] + list(map(int, box_2_frames[-1]))
            else:
                i += 1
                row = [int(frame)] + [int(target_id)] + list(map(int, box_2_frames[frame_num - 1]))

                frame_num += 2

            if data is None:
                data = np.array(row)
            else:
                data = np.vstack((data, np.array(row))).astype(np.int64)
        print(
            'writing video {} target #{} action {}'.format(base_video_name, target_id, action))
        # np.savetxt('{}_{}_before.txt'.format(video_name, target_id), data, fmt='%.2f')
        data = fill_gaps(data, int(target_id))
        # np.savetxt('{}_{}_after.txt'.format(video_name, target_id), data, fmt='%.2f')

        data = smooth(data, window_size=9)

        for detection in data:
            x_c, y_c = detection[2] + detection[4] / 2, detection[3] + detection[5] / 2
            w, h = detection[4], detection[5]
            img = cv2.imread(os.path.join(frame_path, base_video_name, 'img_{:05d}.jpg'.format(
                int(detection[0]))))
            if img is None:
                print('Frame {} Not Found'.format(
                    os.path.join(frame_path,
                                 base_video_name + '_{:06d}.jpg'.format(int(detection[0])))))
                break
            patch = crop_imgs(img, x_c, y_c, [w, h])

            patch = cv2.resize(patch, (32, 64))

            video_writer.write(patch)

        video_writer.release()


if __name__ == '__main__':
    video_dict = load_csv(csv_path)
    filter_video_dict(video_dict, '/home/rvlab/Documents/DRDvideo_processed/AVA_extracted/val/',
                      '/mnt/nasbi/action_recognition/AVA/rawframes/')
