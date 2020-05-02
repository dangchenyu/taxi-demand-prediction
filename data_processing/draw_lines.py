import cv2
import os
def get_video_writer(save_video_path, width, height):
    if save_video_path != '':
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 14, (int(width), int(height)))
    else:
        class MuteVideoWriter():
            def write(self, *args, **kwargs):
                pass

            def release(self):
                pass

        return MuteVideoWriter()
def draw_lines(video_path,save_video,save_frames=True):
    if os.path.isdir(video_path):
        path_list = os.listdir(video_path)
        for video in path_list:
            frame_num=0
            video_base_name = os.path.splitext(video)[0]
            capture = cv2.VideoCapture(os.path.join(video_path, video))
            video_writer = get_video_writer(os.path.join(save_video,  'drawn_videos',video_base_name+'_drawnlines.mp4'),
                                            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                            capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while True:
                ret, frame = capture.read()
                if save_frames:
                    img_save_path=os.path.join(save_video,'raw_frames',video_base_name+'_{:06d}.jpg'.format(frame_num))
                    cv2.imwrite(img_save_path,frame)
                if not ret:
                    break
                frame_num+=1
                cv2.line(frame,(0,360),(1280,360),(0,255,0),1)
                cv2.line(frame, (640, 0), (640, 720), (0, 255, 0), 1)
                video_writer.write(frame)
            video_writer.release()
if __name__ == '__main__':
    draw_lines('/home/rvlab/Documents/DRDvideos/','/home/rvlab/Documents/DRDvideo_processed/')