import cv2
import  numpy as np
from imutils.video import VideoStream
import imutils
from trainModel import recognise_emotion


def face_detection(model, confidence = 0.5):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    print("[*] Strating Video Stream...")
    vs = VideoStream(src=0).start()
    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # grab frame dimension and convert it to a blob
        (h, w) = frame.shape[: 2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass blob through the network and obtain detections & predictions
        net.setInput(blob)
        detections = net.forward()

        # now we loop over the detections, compare the confidence level and
        # draw boxes and confidence values on the screen
        for i in range(0, detections.shape[2]):
            # confidence i.e probailtiy  associated with prediction
            detection_confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if detection_confidence < confidence:
                continue

            # compute (x, y) co-ordinates of the bounding box for object
            box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(int)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            face_area = frame[startY: endY, startX: endX]
            face_area = cv2.cvtColor(face_area, cv2.COLOR_BGR2GRAY)
            face_area = (cv2.resize(face_area, (48, 48)))/127.0
            # ar = face_area / 127.0
            text = recognise_emotion(model, face_area_to_detect_emotion = face_area)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the "q" is pressed, break the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
