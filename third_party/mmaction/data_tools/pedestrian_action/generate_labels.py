import os
import random
def main(frame_path):
    folders=os.listdir(frame_path)
    walking_list=[]
    standing_list=[]
    waving_list=[]
    all_list=[]
    for folder in folders:
        all_list.append(folder)
        if 'walking' in folder:
            walking_list.append(folder)
        if 'standing' in folder:
            standing_list.append(folder)
        if 'waving' in folder:
            waving_list.append(folder)
    train_list=random.sample(walking_list,int(0.8*len(walking_list)))+random.sample(standing_list,int(0.8*len(standing_list)))+random.sample(waving_list,int(0.8*len(waving_list)))
    test_list=list(set(all_list).difference(set(train_list)))
    with open('/home/rvlab/Documents/DRDvideo_processed/annotations/train_action.txt','w+') as o:
        for train_instance in train_list:
            train_imgs_list = os.listdir(frame_path + train_instance)
            train_folder_length=len(train_imgs_list)
            if 'walking' in train_instance:
                action=0
            elif 'standing' in train_instance:
                action=1
            elif 'waving' in train_instance:
                action=2
            o.write('%s %s %s\n'%(str(train_instance),str(train_folder_length),str(action)))
    o.close()
    with open('/home/rvlab/Documents/DRDvideo_processed/annotations/val_action.txt','w+') as w:
        for test_instance in test_list:
            test_imgs_list = os.listdir(frame_path + test_instance)
            test_folder_length=len(test_imgs_list)
            if 'walking' in test_instance:
                action=0
            elif 'standing' in test_instance:
                action=1
            elif 'waving' in test_instance:
                action=2
            w.write('%s %s %s\n'%(str(test_instance),str(test_folder_length),str(action)))
    w.close()
if __name__ == '__main__':
    frame_path='/home/rvlab/Documents/DRDvideo_processed/processed_frames/'
    main(frame_path)