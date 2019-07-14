from mtcnn.mtcnn import MTCNN
import os
import cv2
import numpy as np
def load_image(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=np.asarray(img)
    return img

def extract_face_roi(path):
 img=load_image(path)

 extractor=MTCNN()
 coordinates=extractor.detect_faces(img)
 x1,y1, width, height = coordinates[0]['box']
 x1,y1 = abs(x1), abs(y1)
 x2,y2 = x1+width,y1+height
 face = img[y1:y2, x1:x2]
 img=cv2.resize(face,(160,160))
 img=np.asarray(img)
 return img

def load_faces(path):
    faces=list()

    for file in os.listdir(path):
        final_path=path+file
        face=extract_face_roi(final_path)
        faces.append(face)
    return faces

def load_dataset(path):
    X,y=list(),list()

    for subpath in os.listdir(path):
        final_path=path + subpath + '/'
        if not os.path.isdir(final_path):
            continue

        faces=load_faces(final_path)
        targets=[subpath for _i in range (len(faces))]
        X.extend(faces)
        y.extend(targets)
    return np.asarray(X),np.asarray(y)
