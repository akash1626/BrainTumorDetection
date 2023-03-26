import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical')

image=cv2.imread('D:\\major projct\\archive\\pred')
img=image.fromarray(image)
img=img.resize((64,64))

img=np.array(img)
input_img=np.expand_dims(img, axis=0)
result=model.predict_classes(input_img)

print(result)