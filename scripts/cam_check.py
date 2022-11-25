import cv2

cap = cv2.VideoCapture("rtsp://admin:New@ction2299@10.10.25.31/LiveMedia/ch1/Media1")

while cap.isOpened():

    ret, frame = cap.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)
