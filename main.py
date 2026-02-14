import pickle
import cv2
import os
import face_recognition
import numpy as np

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
video.set(3,640)
video.set(4,480)

img_background = cv2.imread(r"C:\PROJECTS\Face Recognization\Resources\background.png")

# Importing the mode images into a list
mode_path = r"C:\PROJECTS\Face Recognization\Resources\Modes"
mode_folder = os.listdir(mode_path)
img_mode_list = []

for folder in mode_folder:
    img_path = os.path.join(mode_path, folder)
    img_mode_list.append(cv2.imread(img_path))

# print(len(img_mode_list))


#Load the encoding File :
print("......Loading Encoding File.......")
with open("encoding.pkl","rb") as f:
    encoding_list_with_ids = pickle.load(f) 
encoding_list,person_ids = encoding_list_with_ids    
print("......Encoding File Loaded.......")

while True:

    ret, img = video.read()
    if not ret:
        print("Error: Failed to capture image from camera. Check camera index or connection.")
        break


    img_resized = cv2.resize(img,(0,0),None,0.25,0.25)
    img_resized = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)

    Face_current_frame = face_recognition.face_locations(img_resized)
    encode_current_frame = face_recognition.face_encodings(img_resized,Face_current_frame)

    img_background[162:162+480, 55:55+640] = img
    img_background[44:44+633, 808:808+414] = img_mode_list[0]


    for encodeFace,faceloc in zip(encode_current_frame,Face_current_frame):
        matches = face_recognition.compare_faces(encoding_list,encodeFace)
        face_dis = face_recognition.face_distance(encoding_list,encodeFace)
        # print("matches",matches)
        # print("face_dis",face_dis)

        matchIndex = np.argmin(face_dis)
        # print(matchIndex)

        if matches[matchIndex]:
            # print("Known Face")
            name = person_ids[matchIndex]
            # print(name)

            face_cascade = cv2.CascadeClassifier(r"C:\PROJECTS\Face Recognization\haarcascade_frontalface_default.xml")  

            gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)

            face = face_cascade.detectMultiScale(gray,1.1,5)

            for (x,y,w,h) in face:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        else:
            print("Unknown Face")
    cv2.imshow("Background", img_background)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()