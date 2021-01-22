import cv2


#video
video = cv2.VideoCapture('Pedestrians.mp4')

#pre-trained data from cars and pedestrians
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#Loads Pre-trained data from OpenCV library
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


#Runs forever
while True:
    #Reads current frame
    (read_successful,frame) =  video.read()

    if read_successful:
        #Converts colored image to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #Detects objects of different sizes and returns coordinates of a rectangle surrounding the car
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draws rectangle around the cars
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x+1,y+1), (x+w, y+h), (0,255,0),2)
        cv2.rectangle(frame,(x+2,y+2), (x+w, y+h), (0,255,0),2)

    #Draws rectangle around pedestrians
    for(x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,500,500),2)

    #Shows
    cv2.imshow('Street_Tracking', frame) 

    #Waits 1 millisecond for the next frame and keeps going
    key = cv2.waitKey(1)

    #Stops if Q is pressed
    if key== 81 or key == 113:
        break






