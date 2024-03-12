import streamlit as st
import cv2
import requests
import json
import numpy as np
from PIL import Image as im
from lip_detect.solo_vid import lip_detect
import imageio
import base64

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

# Display video if a file is selected
if video_file is not None:
    st.video(video_file)

    vid = video_file.name
    with open(vid, mode='wb') as f:
        f.write(video_file.read()) # save video to disk

    st.write('---- saved to disk ----')

    vidcap = cv2.VideoCapture(vid)
    success = True
    i = 0

    st.write('---- video capturing ----')

    frames = []
    # while vidcap.isOpened():
    while success:
    # while i <= 10:
        success, frame = vidcap.read()
        if frame is not None:
            img = im.fromarray(frame).convert('L')
            lips = lip_detect(np.array(img))
            frames.append(lips.tolist())
            i += 1
            if i % 10 == 0:
                st.image(lips)
                st.write(f'---- frame {i} complete ----')

    st.write('---- releasing video capture ----')

    vidcap.release()

    st.write('---- all frames captured ----')

    st.write('---- creating lip gif ----')

    gif_list = frames[:75]  # Example list of 10 random frames
    # Save the frames as a GIF
    imageio.mimsave('name.gif', gif_list, duration=0.1)

    file_ = open("name.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="lips gif">',
        unsafe_allow_html=True,
)

    st.write('---- posting request ----')

    st.write(type(frames))

    response = requests.post("http://127.0.0.1:8000/predict", json=json.dumps(frames))
    st.write('---- post request sent ----')

    st.write(response)

    result = response.json()['data']

    first_frame = np.array(json.loads(result))

    st.write(first_frame.shape)
    st.image(first_frame)




    # # Save uploaded file to temporary location
    # temp_dir = tempfile.TemporaryDirectory()
    # temp_file_path = os.path.join(temp_dir.name, video_file.name)
    # with open(temp_file_path, "wb") as f:
    #     f.write(video_file.read())

    # output_file = "current_vid/vid_input.mp4"
    # command = ["deface", temp_file_path, "-o", output_file]

    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = process.communicate()

    # if process.returncode == 0:
    #     st.success("Deface process completed successfully!")
    #     st.video(output_file)

    #     # Download code
    #     with open('current_vid/vid_input.mp4', 'rb') as f:
    #         st.download_button('Download MP4', f, file_name='current_vid/vid_input.mp4')
    #         os.remove(output_file)
    # else:
    #     st.error("Deface process failed. Error message:")
    #     st.code(stderr.decode("utf-8"))

    # # Cleanup temporary directory
    # temp_dir.cleanup()






# if video_selection:
#     video_path = os.path.join(directory_path, video_selection)

#     capture = cv2.VideoCapture(video_path)
#     placeholder = st.empty()

#     while capture.isOpened():
#         ret, frame = capture.read()
#         if ret:
#             placeholder.image(frame, channels="BGR", width = 800)
#         else:
#             break
#     capture.release()
