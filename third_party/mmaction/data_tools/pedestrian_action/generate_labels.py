import os

def main(frame_path):
    folders=os.listdir(frame_path)
    with open('/home/rvlab/Documents/hmdb_tracked/train.txt','w+') as w:
        for folder in folders:
            imgs_list=os.listdir(frame_path+folder)
            folder_length=len(imgs_list)
            if 'walk' in folder:
                action=0
            else:
                action=1
            w.write('%s %s %s\n'%(str(folder),str(folder_length),str(action)))
    w.close()

if __name__ == '__main__':
    frame_path='/home/rvlab/Documents/hmdb_tracked/processed_frames/'
    main(frame_path)