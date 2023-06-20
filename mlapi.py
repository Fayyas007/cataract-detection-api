from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import urllib.request as urllib
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import tensorflow as tf

app = FastAPI()

class ScoringItem(BaseModel):
    link: str

new_model = load_model('cataracttest.h5')

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    # df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    image = item.link
    image3 = image_url_to_numpy_array_urllib(image,format=None)
    resize = tf.image.resize(image3, (256,256))
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    print(yhat[0][0])
    pred = float(yhat[0][0])
    return {"cataract" : pred}
    

def image_url_to_numpy_array_urllib(url,format=None):
    ## read as HTTPResponse 
    resp = urllib.urlopen(url)
    ## read as 1D bytearray
    resp_byte_array = resp.read()
    ## returns a bytearray object which is a mutable sequence of integers in the range 0 <=x< 256
    mutable_byte_array = bytearray(resp_byte_array)
    ## read as unsigned integer 1D numpy array
    image = np.asarray(mutable_byte_array, dtype="uint8")
    ## To decode the 1D image array into a 2D format with RGB color components we make a call to cv2.imdecode
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if format=='BGR' :
        ## return BGR format array
        return image
    ## cv2.imdecode converted array into BGR format , convert it to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return the image
    return image