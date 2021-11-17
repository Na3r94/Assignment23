# This Python file uses the following encoding: utf-8

import time
import sys
import random
import cv2
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import *
from PySide6.QtCore import QThread, Signal

class Main(QWidget):
    def __init__(self):
        super(Main, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load("form.ui")
        self.ui.btn_sticker.clicked.connect(self.stickerr)
        self.ui.btn_eye.clicked.connect(self.eyelip)
        self.ui.btn_face.clicked.connect(self.pixel)
        self.ui.btn_effect.clicked.connect(self.filter)

        self.ui.show()

    def stickerr(self):

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        my_video = cv2.VideoCapture(0)
        sticker1 = cv2.imread('sticker1.png', cv2.IMREAD_UNCHANGED)
        sticker2 = cv2.imread('sticker2.png', cv2.IMREAD_UNCHANGED)

        sticker1_image = sticker1[:, :, 0:3]
        sticker1_image_gray = cv2.cvtColor(sticker1_image, cv2.COLOR_RGB2GRAY)
        sticker1_mask = sticker1[:, :, 3]
        while True:
            validation, frame = my_video.read()
            if validation is not True:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            faces = face_detector.detectMultiScale(frame_gray, 1.3)

            for face in faces:
                x, y, w, h = face

                image_face = frame_gray[y:y+h, x:x+h]
                sticker1_image_gray_resized = cv2.resize(sticker1_image_gray, (w,h))
                sticker1_mask_gray_resized = cv2.resize(sticker1_mask, (w,h))

                sticker1_image_gray_resized = sticker1_image_gray_resized.astype(float) / 255
                sticker1_mask_gray_resized = sticker1_mask_gray_resized.astype(float) / 255
                image_face = image_face.astype(float) / 255

                foreground = cv2.multiply(sticker1_image_gray_resized, sticker1_mask_gray_resized)

                background = cv2.multiply(image_face, 1 - sticker1_mask_gray_resized)

                result = cv2.add(foreground, background)

                result *= 255

                frame_gray[y:y + h, x:x + w] = result
            cv2.imshow('output', frame_gray)
            cv2.waitKey(1)


    def eyelip(self):
        eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
        lip_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        my_video = cv2.VideoCapture(0)
        eye_sticker = cv2.imread('eye.png', cv2.IMREAD_UNCHANGED)
        lips_sticker = cv2.imread('lips.png', cv2.IMREAD_UNCHANGED)

        eye_image = eye_sticker[:, :, 0:3]
        eye_image_gray = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)
        eye_mask = eye_sticker[:, :, 3]

        lips_image = lips_sticker[:, :, 0:3]
        lips_image_gray = cv2.cvtColor(lips_image, cv2.COLOR_RGB2GRAY)
        lips_mask = lips_sticker[:, :, 3]


        while True:
            validation, frame = my_video.read()
            if validation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            eyes = eye_detector.detectMultiScale(frame, 1.6, 15)
            lips = lip_detector.detectMultiScale(frame, 1.5, 20)

            for eye in eyes:
                ex, ey, ew, eh = eye
                image_face = frame_gray[ey:ey+eh, ex:ex+ew]

                eye_sticker_gray_resized = cv2.resize(eye_image_gray, (ew, eh))
                eye_mask_gray_resized = cv2.resize(eye_mask, (ew, eh))

                eye_sticker_gray_resized = eye_sticker_gray_resized.astype(float) / 255
                eye_mask_gray_resized = eye_mask_gray_resized.astype(float) / 255
                image_face = image_face.astype(float) / 255


                eye_foreground = cv2.multiply(eye_sticker_gray_resized, eye_mask_gray_resized)

                eye_background = cv2.multiply(image_face, 1 - eye_mask_gray_resized)

                result = cv2.add(eye_foreground, eye_background)

                result *= 255
                frame_gray[ey:ey + eh, ex:ex + ew] = result

            for lip in lips:
                lx, ly, lw, lh = lip
                image_face = frame_gray[ly:ly+lh, lx:lx+lw]

                lip_sticker_gray_resized = cv2.resize(lips_image_gray, (lw, lh))
                lip_mask_gray_resized = cv2.resize(lips_mask, (lw, lh))

                lip_sticker_gray_resized = lip_sticker_gray_resized.astype(float) / 255
                lip_mask_gray_resized = lip_mask_gray_resized.astype(float) / 255
                image_face = image_face.astype(float) / 255


                lips_foreground = cv2.multiply(lip_sticker_gray_resized, lip_mask_gray_resized)

                lips_background = cv2.multiply(image_face, 1 - lip_mask_gray_resized)

                result = cv2.add(lips_foreground, lips_background)

                result *= 255

                frame_gray[ly:ly + lh, lx:lx + lw] = result

            cv2.imshow('output1', frame_gray)
            cv2.waitKey(1)


    def pixel(self):
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        my_video = cv2.VideoCapture(0)

        while True:

            validation, frame = my_video.read()

            if validation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            faces = face_detector.detectMultiScale(frame_gray, 1.3)

            for face in faces:
                x, y, w, h = face

                wid, hei = (8, 8)
                temp = cv2.resize(frame_gray[y:y + h, x:x + w], (wid, hei), interpolation=cv2.INTER_LINEAR)
                output = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

                frame_gray[y:y + h, x:x + w] = output

            cv2.imshow('output', frame_gray)
            cv2.waitKey(10)

    def filter(self):
        my_video = cv2.VideoCapture(0)

        def apply_invert(frame):
            return cv2.bitwise_not(frame)
        while True:

            validation, frame = my_video.read()

            if validation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            invert = apply_invert(frame_gray)

            cv2.imshow('output', invert)
            cv2.waitKey(10)

    def flip(self):
        my_video = cv2.VideoCapture(0)

        while True:

            validation, frame = my_video.read()

            if validation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            rows, cols = frame_gray.shape
            half_frame = frame_gray[:, 0:cols // 2]

            flipped_half_frame = cv2.flip(half_frame, 1)

            frame_gray[:, cols // 2:] = flipped_half_frame
            cv2.imshow('output', frame_gray)
            cv2.waitKey(3)

if __name__ == "__main__":
    app = QApplication([])
    widget = Main()
    sys.exit(app.exec_())