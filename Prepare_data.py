from utils import load_dataset
import pickle
import os

train_path='C:/Users/ADMIN/Desktop/Facenet/Celebrity_dataset/train/'
test_path='C:/Users/ADMIN/Desktop/Facenet/Celebrity_dataset/val/'
save_path='C:/Users/ADMIN/Desktop/Facenet/models_pickle_dumps/'

X_train,Y_train=load_dataset(train_path)
X_test,Y_test=load_dataset(test_path)

with open(os.path.join(save_path,"train.pickle"),"wb") as f:
    pickle.dump((X_train,Y_train),f)

with open(os.path.join(save_path,"val.pickle"),"wb") as f:
    pickle.dump((X_test,Y_test),f)

print("Done")
