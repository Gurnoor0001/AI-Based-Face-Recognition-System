from copyreg import pickle
import face_recognition
import cv2
import face_recognition
import os
import pickle

# Importing the person images into a list
images_path = r"C:\PROJECTS\Face Recognization\Images"
images_folder = os.listdir(images_path)
img_list = []
person_ids = []


for folder in images_folder:
    img_path = os.path.join(images_path, folder)
    img_list.append(cv2.imread(img_path))
    person_ids.append(os.path.splitext(folder)[0])

# print(person_ids)
# print(len(img_list))


# function togenerate encodings

def encoding(img_list):
    encoding = []
    for img in img_list:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoding.append(encode)

    return encoding
print(10*"-","Encoding Start",10*"-")
encoding_list = encoding(img_list)
encoding_list_with_ids = [encoding_list,person_ids]
    
print(10*"-","Encoding Ends",10*"-")

with open("encoding.pkl","wb") as f :
    pickle.dump(encoding_list_with_ids,f)
