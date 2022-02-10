import asyncio
import os
import os.path


import cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

from av import video

from imageai.Detection import VideoObjectDetection

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaBlackhole

# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
#
# from starlette.requests import Request
# from starlette.responses import HTMLResponse
# from starlette.templating import Jinja2Templates


# dir = os.path.abspath(os.curdir)
# print(dir)
# rtsp = cv2.VideoCapture('192.168.0.120:8004/live/main')

RTSP_URL = "rtsp://admin:38912082Au@192.168.0.120:8004/live/main"
VIDEO = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
# Обученные XML-классификаторы описывают некоторые особенности некоторого объекта, который мы хотим обнаружить
car_cascade = cv2.CascadeClassifier(r'/home/maksim/PycharmProjects/anpr/ctv/carx.xml')

while(1):

    ret, frame = VIDEO.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)