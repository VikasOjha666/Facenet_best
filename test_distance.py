import cv2
import numpy as np
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from utils import extract_face_roi
import pickle
from scipy.spatial.distance import cosine

model=load_model('facenet_keras.h5')

name_array=['ben_afflek','elton_john','jerry_seinfeld','madonna','mindy_kaling']

def get_embedding(model,face):
    face=face.astype('float32')
    mean,std=face.mean(),face.std()
    face=(face-mean)/std
    face=np.expand_dims(face,axis=0)
    embedding=model.predict(face)
    return embedding[0]

def findEuclideanDistance(source_representation, test_representation):
 euclidean_distance = source_representation - test_representation
 euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
 euclidean_distance = np.sqrt(euclidean_distance)
 return euclidean_distance

Normaliser = Normalizer(norm='l2')


img=extract_face_roi('Celebrity_dataset/val/jerry_seinfeld/httpblognjcomentertainmentimpactcelebritiesmediumjerrybjpg.jpg')

embedding=get_embedding(model,img)
embedding=np.reshape(embedding,(-1,2))
norm_embedding=Normaliser.transform(embedding)
norm_embedding=np.reshape(norm_embedding,(128,))

img2=extract_face_roi('Celebrity_dataset/val/ben_afflek/httpafilesbiographycomimageuploadcfillcssrgbdprgfacehqwMTENDgMDUODczNDcNTcjpg.jpg')

embedding2=get_embedding(model,img2)
embedding2=np.reshape(embedding2,(-1,2))
norm_embedding2=Normaliser.transform(embedding2)
norm_embedding2=np.reshape(norm_embedding2,(128,))

score=cosine(norm_embedding,norm_embedding2)
print('Score=',score)
if score<0.65:
    print('Similar')
else:
    print('Not Similar')
