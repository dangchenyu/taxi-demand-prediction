import cv2
from .encode import Encoder


class ImagePatchEncoder(Encoder):
    def __init__(self, resize_to=(32, 32)):
        super(ImagePatchEncoder, self).__init__()
        self.resize_to = resize_to
        self.name = 'patch'

    def __call__(self, detections, img):
        imgs = []
        for detection in detections:
            box = detection.box
            patch = self.crop_imgs(img, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, [box[2] - box[0], box[3] - box[1]])
            patch = cv2.resize(patch, self.resize_to)
            imgs.append(patch)
        return imgs

    @staticmethod
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
