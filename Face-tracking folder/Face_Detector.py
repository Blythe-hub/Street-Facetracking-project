import cv2
import random

#Loads Pre-trained data from OpenCV library
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#img = cv2.imread('face.png')

webcam = cv2.VideoCapture(0)


#Iterates over frames
while True:

    #Reads every current frame
    successful_frame_read, frame = webcam.read()

    #Converts colored image to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detects objects of different sizes and returns coordinates of a rectangle surrounding the face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draws rectangle around the faces
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (127,255,60,5))

    #Shows
    cv2.imshow('Face_Detector', frame) 

    #Waits 1 millisecond for the next frame and keepsw going
    key = cv2.waitKey(1)

    #Stops if Q is pressed
    if key== 81 or key == 113:
        break

#Stops webcam
webcam.release()







