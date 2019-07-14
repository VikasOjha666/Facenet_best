import cv2
import numpy as np
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from utils import extract_face_roi
import pickle

model=load_model('facenet_keras.h5')

name_array=['ben_afflek','elton_john','jerry_seinfeld','madonna','mindy_kaling']

def get_embedding(model,face):
    face=face.astype('float32')
    mean,std=face.mean(),face.std()
    face=(face-mean)/std
    face=np.expand_dims(face,axis=0)
    embedding=model.predict(face)
    return embedding[0]

Normaliser = Normalizer(norm='l2')

#img=cv2.imread()
img=extract_face_roi('Celebrity_dataset/val/madonna/httpcdncdnjustjaredcomwpcontentuploadsheadlinesmadonnatalksparisattackstearsjpg.jpg')
embedding=get_embedding(model,img)
embedding=np.reshape(embedding,(-1,2))
norm_embedding=Normaliser.transform(embedding)
norm_embedding=np.reshape(norm_embedding,(1,128))

svc_model=open('SVC_model.pkl','rb')
svc_model=pickle.load(svc_model)

predicted=svc_model.predict(norm_embedding)
print('Prediction:',name_array[int(predicted)])
