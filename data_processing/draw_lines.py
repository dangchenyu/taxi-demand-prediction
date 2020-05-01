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
def main(video_path,save_video):
    if os.path.isdir(video_path):
        path_list = os.listdir(video_path)
        for video in path_list:
            video_base_name = os.path.splitext(video)[0]
            capture = cv2.VideoCapture(os.path.join(video_path, video))
            video_writer = get_video_writer(os.path.join(save_video,  video),
                                            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                            capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        capture = cv2.VideoCapture(os.path.join(video_path))
        video_writer = get_video_writer(os.path.join(save_video),
                                        capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                        capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = capture.read()

            if not ret:
                break
            cv2.line(frame,(0,360),(1280,360),(0,255,0),1)
            cv2.line(frame, (640, 0), (640, 720), (0, 255, 0), 1)
            video_writer.write(frame)
        video_writer.release()
if __name__ == '__main__':
    main('/home/rvlab/Desktop/M19040313550500131.mp4','/home/rvlab/Desktop/drawn_lines.mp4')