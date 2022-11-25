import cv2

cap = cv2.VideoCapture("rtsp://admin:New@ction2299@10.10.25.31/LiveMedia/ch1/Media1")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print(frame_width)
print(frame_height)
out = cv2.VideoWriter('walkpose.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    out.write(frame)

    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
