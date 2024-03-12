from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import io
import cv2
#from fastapi.responses import JSONResponse
#from app.modelutil import load_model
import numpy as np
import json

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/home/")
def home():
    return {'message': "Hello"}

@app.post("/predict/")
async def numpy_test(test: Request):
    data = await test.json()
    print('oh my god we did it?!')
    first_frame = np.array(np.array(json.loads(data))[0])
    print(first_frame.shape)
    return {"message": "Received data for prediction", "data": json.dumps(first_frame.tolist())}
