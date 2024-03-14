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

    vidcap = cv2.VideoCapture(vid)
    success = True
    i = 0

    frames = []
    while success:
        # while vidcap.isOpened():
    # while i <= 10:
        success, frame = vidcap.read()
        if frame is not None:
            lips = lip_detect(np.array(im.fromarray(frame).convert('L')))
            frames.append(lips.tolist())
            i += 1
            if i % 10 == 0:
                response = requests.post("https://lip-reader-docker-zn34um6luq-nw.a.run.app/send_frames/", json=json.dumps(frames))
                st.write(response.json())
                if response.ok:
                    frames = []
                else:
                    st.write(response)
                st.write(f'---- frame {i} complete {str(response)} ----')

    vidcap.release()

    response = requests.post("https://lip-reader-docker-zn34um6luq-nw.a.run.app/send_frames/", json=json.dumps(frames))

    st.write(response)

    st.image(lips)

    # AWAITING CORRECT API LINK
    prediction = requests.get("https://lip-reader-docker-zn34um6luq-nw.a.run.app/predict/")

    if prediction.ok:
        st.balloons()
        st.write(prediction.json())
    else:
        st.balloons()
        st.write('I AM A CORRECT RESPONSE PLACEHOLDER')
