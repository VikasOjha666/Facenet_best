from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import os
import pickle
import numpy as np

save_path='C:/Users/ADMIN/Desktop/Facenet/models_pickle_dumps/'
model=load_model('facenet_keras.h5')

with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (X_train,Y_train) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (X_test,Y_test) = pickle.load(f)

def get_embedding(model,face):
    face=face.astype('float32')
    mean,std=face.mean(),face.std()
    face=(face-mean)/std
    face=np.expand_dims(face,axis=0)
    embedding=model.predict(face)
    return embedding[0]

train_embedding=[]
test_embedding=[]

for face in X_train:
    embedding=get_embedding(model,face)
    train_embedding.append(embedding)

train_embedding=np.asarray(train_embedding)

for face in X_train:
    embedding=get_embedding(model,face)
    test_embedding.append(embedding)

test_embedding=np.asarray(test_embedding)

with open(os.path.join(save_path,"train_embedding.pickle"),"wb") as f:
    pickle.dump((train_embedding,Y_train),f)

with open(os.path.join(save_path,"test_embedding.pickle"),"wb") as f:
    pickle.dump((test_embedding,Y_test),f)


with open(os.path.join(save_path, "train_embedding.pickle"), "rb") as f:
    (X_train,Y_train) = pickle.load(f)

with open(os.path.join(save_path, "test_embedding.pickle"), "rb") as f:
    (X_test,Y_test) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (X_test_pix,Y_test_pix) = pickle.load(f)

Normaliser = Normalizer(norm='l2')
X_train = Normaliser.transform(X_train)
X_test = Normaliser.transform(X_test)

one_hot_enocder = LabelEncoder()
one_hot_enocder.fit(Y_train)
Y_train = one_hot_enocder.transform(Y_train)
Y_test = one_hot_enocder.transform(Y_test)

SVC_model = SVC(kernel='linear', probability=True)
SVC_model.fit(X_train, Y_train)

model_name='SVC_model.pkl'
model_pkl=open(model_name,'wb')
pickle.dump(SVC_model,model_pkl)
model_pkl.close()
