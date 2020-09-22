import os
import cv2
import logging
import argparse
import numpy as np


def fill_gaps(track, target_id):
    logging.info('Gap-filling: Length before filling: {}'.format(len(track)))
    output_trajectory = []
    current_frame = -1
    for i in range(len(track)):
        if current_frame > -1 and track[i][0] - current_frame > 1:
            box_gap = track[i][2:6] - output_trajectory[-1][2:6]
            frame_gap = track[i][0] - current_frame
            logging.info('Gap found between frame {} and frame {}'.format(current_frame, track[i][0]))
            unit = box_gap / frame_gap
            logging.info('Gap unit: ({:.3f}, {:.3f}, {:.3f}, {:.3f})'.format(unit[0], unit[1], unit[2], unit[3]))
            for j in range(current_frame + 1, int(track[i][0])):
                to_fill = output_trajectory[-1][2:6] + unit
                output_trajectory.append(np.array([j, target_id, to_fill[0], to_fill[1], to_fill[2], to_fill[3]]))
        current_frame = int(track[i][0])
        output_trajectory.append(track[i])
    logging.info('Gap-filling: After filling: {}'.format(len(output_trajectory)))
    return np.array(output_trajectory)


def smooth(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    # Padding
    smoothed = data[:, 2:]
    pad_left = smoothed[:(window_size // 2), :][::-1, :]
    pad_right = smoothed[-(window_size // 2):, :][::-1, :]
    smoothed = np.concatenate([pad_left, smoothed, pad_right], axis=0)
    smoothed = np.array([np.convolve(smoothed[:, i], window, mode='valid') for i in range(4)]).transpose(1, 0)
    try:
        data = np.concatenate((data[:, 0:2], smoothed), axis=1)
    except:
        data = data
    return data


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
           int(x_base + x_c - half_w): int(x_base + x_c + half_w+half_w*0.5), :]


def crop(img, x_c, y_c, window_size):
    x_base = 0
    y_base = 0
    padded_img = img
    half_window = int(window_size // 2)
    if x_c < half_window:
        padded_img = cv2.copyMakeBorder(img, 0, 0, half_window, 0, borderType=cv2.BORDER_REFLECT)
        x_base = half_window
    if x_c > img.shape[1] - half_window:
        padded_img = cv2.copyMakeBorder(img, 0, 0, 0, half_window, borderType=cv2.BORDER_REFLECT)
    if y_c < half_window:
        padded_img = cv2.copyMakeBorder(img, half_window, 0, 0, 0, borderType=cv2.BORDER_REFLECT)
        y_base = half_window
    if y_c > img.shape[0] - half_window:
        padded_img = cv2.copyMakeBorder(img, 0, half_window, 0, 0, borderType=cv2.BORDER_REFLECT)

    return padded_img[int(y_base + y_c - half_window): int(y_base + y_c + half_window),
           int(x_base + x_c - half_window): int(x_base + x_c + half_window), :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video patch cropper')
    parser.add_argument('--annotation_file', type=str, help='Path to the annotation file')
    parser.add_argument('--video_dir', type=str, help='Path to the extracted raw videos')
    parser.add_argument('--track_dir', type=str, help='Path to the tracked trajectory files')
    parser.add_argument('--output_dir', type=str, help='Path to the output video patches')
    parser.add_argument('--video_size', default=(64, 128), type=tuple, required=False)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    txt_list = os.listdir(args.track_dir)
    video_name = 'campus_data'

    cap = cv2.VideoCapture(os.path.join(args.video_dir, video_name + '.mp4'))

    for txt in txt_list:
        # video_name=os.path.splitext(txt)[0][:18]0

        data = np.loadtxt(os.path.join(args.track_dir, txt), delimiter=',')
        data = data[:, :6]
        if '6-10' in txt:
            action = 'stand_look'
        elif '1-5' in txt:
            action = 'wave'
        elif '11-15' in txt:
            action = 'stand'
        elif '16-20' in txt:
            action = 'walk'
        elif 'others' in txt:
            action = 'undefined'
        ids = np.unique(data[:, 1])
        for target_id in ids:
            target_id = int(target_id)
            data_cur = data[data[:, 1] == target_id]
            start_frame = int(data_cur[0][0])
            end_frame = int(data_cur[-1][0])

            video_writer = cv2.VideoWriter(
                os.path.join(args.output_dir,action,
                             '{}_{}_{}_{}.mp4'.format(os.path.splitext(video_name)[0], target_id, start_frame,
                                                      end_frame)),
                cv2.VideoWriter_fourcc(*'mp4v'), 15, (64, 128))

            logging.info('Filling gaps for video {} target #{}'.format(video_name, target_id))
            # np.savetxt('{}_{}_before.txt'.format(video_name, target_id), data, fmt='%.2f')
            data_cur = fill_gaps(data_cur, target_id)
            # np.savetxt('{}_{}_after.txt'.format(video_name, target_id), data_cur, fmt='%.2f')

            data_cur = smooth(data_cur, window_size=9)
            for detection in range(0,len(data_cur[:-2,])+1,2):
                box = data_cur[detection][2:6]
                x_c, y_c = (data_cur[detection][2] + data_cur[detection][4] / 2), (data_cur[detection][3] + data_cur[detection][5] / 2)
                w, h = data_cur[detection][4], data_cur[detection][5]

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(data_cur[detection][0]))
                a, img = cap.read()
                # cv2.rectangle(img,(int(detection[2]),int(detection[3])),(int(detection[2]+w),int(detection[3]+h)),(0,255,0))
                # cv2.imshow('test',img)
                # cv2.waitKey(0)
                if img is None:
                    print('Frame {} Not Found'.format(
                        os.path.join(args.rawframes_dir, video_name, '{}.png'.format(int(data_cur[detection][0])))))
                    break
                patch = crop_imgs(img, x_c, y_c, [w, h])

                patch = cv2.resize(patch, (args.video_size[0], args.video_size[1]))
                video_writer.write(patch)
            print('writing: ', video_name + str(target_id))
            video_writer.release()
