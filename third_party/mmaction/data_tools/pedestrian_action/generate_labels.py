import os
import random
def main(base_path,val):
    folder_list=os.listdir(base_path)
    train_folder_list=[]
    val_folder_list=[]
    for folder in folder_list:
        if 'campus' in folder:
            if val not in folder:
                train_folder_list.append(folder)
            else:
                val_folder_list.append(folder)
    walking_list=[]
    standing_list=[]
    i=0
    for folder in train_folder_list:
        if 'walk' in folder:
            walking_list.append(folder)
        if 'stand' in folder:
            standing_list.append(folder)
        if 'wave' in folder:
            standing_list.append(folder)
    standing_list=random.sample(standing_list,int(len(standing_list)/2))
    train_list=walking_list+standing_list

    # with open('/home/rvlab/Documents/DRDvideo_processed/AVA_annotation/train_action.txt','w+') as o:
    with open('/home/rvlab/Documents/DRDvideo_processed/annotations/train_action'+val+'.txt','w+') as o:

        for train_instance in train_list:
            train_imgs_list = os.listdir(base_path + train_instance)
            train_folder_length=len(train_imgs_list)
            if 'walk' in train_instance:
                action=0
            elif 'stand' in train_instance:
                action=1
            elif 'wave' in train_instance:
                action=1
            o.write('%s %s %s\n'%(str(train_instance),str(train_folder_length),str(action)))
    o.close()
    walking_list=[]
    standing_list=[]
    for folder in val_folder_list:
        if 'walk' in folder:
            walking_list.append(folder)
        if 'stand' in folder:
            standing_list.append(folder)
        if 'wave' in folder:
            standing_list.append(folder)
    val_list = walking_list + standing_list
    with open('/home/rvlab/Documents/DRDvideo_processed/annotations/val_action'+val+'.txt', 'w+') as w:

        for test_instance in val_list:
                test_imgs_list = os.listdir(base_path + test_instance)
                test_folder_length=len(test_imgs_list)
                if 'walk' in test_instance:
                    action=0
                elif 'stand' in test_instance:
                    action=1
                elif 'wave'  in test_instance:
                    action=1
                w.write('%s %s %s\n'%(str(test_instance),str(test_folder_length),str(action)))
    w.close()
import os
import random
def temp(frame_path,val):
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
    train_list=random.sample(walking_list,int(0.5*len(walking_list)))+random.sample(standing_list,int(0.5*len(standing_list)))+random.sample(waving_list,int(0.5*len(waving_list)))
    test_list=list(set(all_list).difference(set(train_list)))
    # with open('/home/rvlab/Documents/DRDvideo_processed/AVA_annotation/train_action.txt','w+') as o:
    with open('/home/rvlab/Documents/DRDvideo_processed/annotations/train_action'+val+'.txt','w+') as o:

        for train_instance in train_list:
            train_imgs_list = os.listdir(frame_path + train_instance)
            train_folder_length=len(train_imgs_list)
            if 'walking' in train_instance:
                action=0
            elif 'standing' in train_instance:
                action=1
            elif 'waving' in train_instance:
                action=1
            o.write('%s %s %s\n'%(str(train_instance),str(train_folder_length),str(action)))
    o.close()
    # with open('/home/rvlab/Documents/DRDvideo_processed/AVA_annotation/val_action.txt','w+') as w:
    with open('/home/rvlab/Documents/DRDvideo_processed/annotations/val_action'+val+'.txt', 'w+') as w:

        for test_instance in test_list:
                test_imgs_list = os.listdir(frame_path + test_instance)
                test_folder_length=len(test_imgs_list)
                if 'walking' in test_instance:
                    action=0
                elif 'standing' in test_instance:
                    action=1
                elif 'waving'  in test_instance:
                    action=1
                w.write('%s %s %s\n'%(str(test_instance),str(test_folder_length),str(action)))
    w.close()

if __name__ == '__main__':
    # frame_path='/home/rvlab/Documents/DRDvideo_processed/processed/'#DRD
    base_path='/home/rvlab/Documents/DRDvideo_processed/campus_processed/'#combined
    # frame_path = '/home/rvlab/Documents/DRDvideo_processed/AVA_processed/'
    main(base_path,val='5-8')