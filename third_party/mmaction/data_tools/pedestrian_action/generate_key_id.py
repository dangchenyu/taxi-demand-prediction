import os
import glob


def main(data_path):
    txt_list = glob.glob(data_path + '/*/*.txt')
    with open(data_path + 'label.txt', 'w+') as w:

        for item in txt_list:
            id_dict = {}
            basename = '/mnt/nasbi/action_recognition/HMDB51/videos/' + \
                       os.path.splitext(item)[0].split('hmdb_tracked/')[1]
            f = open(item)
            tracked_info = f.readlines()
            for line in tracked_info:
                line_list = line.split(',')

                if line_list[1] not in id_dict.keys():
                    id_dict[line_list[1]] = [1, [line_list[0], 0]]
                else:
                    id_dict[line_list[1]][0] = id_dict[line_list[1]][0] + 1
                    id_dict[line_list[1]][1][1] = line_list[0]
            new_dict = {v[0]: k for k, v in id_dict.items()}
            try:
                max_id_ori = max(id_dict.values())
                max_id = new_dict[max_id_ori[0]]
                start_frame, end_frame = max_id_ori[1]
                if int(end_frame) - int(start_frame) > 60:
                    w.write("{},{},{},{}\n".format((basename + '.avi'), max_id, start_frame, end_frame))
                f.close()
            except ValueError:
                continue
    w.close()


if __name__ == '__main__':
    data_path = '/home/rvlab/Documents/hmdb_tracked/'
    main(data_path)
