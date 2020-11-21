# machine learning using openCv/ same eye detection can be done
import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    # change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects where are the faces
    faces = face_detector.detectMultiScale(frame_grayscale)

    # scale factor actually blurs the image for getting the face , minNeighbors minimum about of neighors of small smiles that can be it's neighbors

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Get sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # # find all smiles in face
        # for(x_, y_, w_, h_) in smiles:

        #     cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_),
        #                   (50, 50, 200), 4)

        if(len(smiles)) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=4,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    cv2.imshow('Smile Detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


webcam.release()
cv2.destroyAllWindows()

###.....Project Completed......###
