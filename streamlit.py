import streamlit as st
import cv2
import requests
import json
import numpy as np
from PIL import Image as im
from lip_detect.solo_vid import lip_detect

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

# Display video if a file is selected
if video_file is not None:
    st.video(video_file)

    vid = video_file.name
    with open(vid, mode='wb') as f:
        f.write(video_file.read()) # save video to disk

    st.write('---- saved to disk ----')
    container = st.empty()
    clock = f"{0:02d}:{0:02d}"
    st.write('---- clock starts ----')

    vidcap = cv2.VideoCapture(vid)
    success = True
    i = 0

    st.write('---- video capturing ----')

    frames = []
    while vidcap.isOpened():
        for secs in range(0, 1000, 1):
            mm, ss = secs // 60, secs % 60
            container.metric("Time Lapsed", f"{mm:02d}:{ss:02d}")

    while success:
    # while i <= 10:
        success, frame = vidcap.read()
        if frame is not None:
            img = im.fromarray(frame).convert('L')
            lips = lip_detect(np.array(img))
            frames.append(lips.tolist())
            i += 1
            st.write(f'---- frame {i} complete ----')

    st.write('---- clock ends ----')
    container.metric("Time Lapsed", clock)

    st.image(lips)

    vidcap.release()

    st.write('---- all frames captured ----')

    # AWAITING CORRECT API LINK
    response = requests.post("http://127.0.0.1:8000/predict", json=json.dumps(frames))

    st.write('---- post request sent ----')

    st.write(response)

    result = response.json()['data']

    first_frame = np.array(json.loads(result))

    st.write(first_frame.shape)
    st.image(first_frame)
