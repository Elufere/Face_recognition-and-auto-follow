import cv2
import numpy as np
import face_recognition
import os
from djitellopy import tello
import time

encodingList = []
root_path = 'venv/Resources/Images'
images = os.listdir(root_path)

for image in images:
    im = face_recognition.load_image_file(os.path.join(root_path, image))

    encodings = face_recognition.face_encodings(im)
    if len(encodings) == 0:
        continue
    encodingList.append(encodings[0])

print(len(encodingList))

# vid = cv2.VideoCapture(0)


me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 25, 0)
time.sleep(2.2)

w, h = 360, 240
fbRange = [2000, 5000]
pid = [0.2, 0.2, 0]
pError = 0

# Load the image of the face to track and encode it
# face_to_track = face_recognition.load_image_file('venv/Resources/Images/Dre.jpg')
# face_to_track_encodings = face_recognition.face_encodings(face_to_track)[0]




def findEncodedFace(img):
    # small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    # rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(img)
    # Encode each detected face and compare it with the reference face
    print("faces found: ", len(face_locations))
    for face_location in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top, right, bottom, left = face_location
        #
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4

        face_encodings = face_recognition.face_encodings(img, [(top, right, bottom, left)])
        # print(len(face_encodings[0]))
        # Compare the face encoding with the reference face encoding

        # img_ = cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        # img_ = cv2.cvtColor(img, cv2.BGR2RGB)


        # w_ = right - left
        # h_ = bottom - top
        # cx = left + w_ // 2
        # cy = top + h_ // 2
        # return img_, [[cx, cy], w_ * h_]
        if len(face_encodings) > 0 and any(face_recognition.compare_faces(encodingList, face_encodings[0])):
            img_ = cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            w_ = right - left
            h_ = bottom - top
            cx = left + w_ // 2
            cy = top + h_ // 2
            return img_, [[cx, cy], w_ * h_]
    return img, [[0, 0], 0]


def trackEncodedFace(img, info, w, pid, pError):
    # print(info)

    x = info[0][0]
    y = info[0][1]
    area = info[1]
    fb = 0
    error = x - w // 2

    if abs(error) < 10:
        error = 0

    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    threshold = 200
    if fbRange[0] < area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        diff = abs(area - fbRange[1])
        if diff > threshold:
            fb = -20
        else:
            fb = -10
    elif area < fbRange[0] and area != 0:
        diff = abs(area - fbRange[0])
        if diff > threshold:
            fb = 20
        else:
            fb = 10
    if x == 0:
        speed = 0
        error = 0
    me.send_rc_control(0, fb, 0, speed)
    if x:
        print(area, fb, speed)
    return error


while True:
    img = me.get_frame_read().frame
    # ret, img = vid.read()
    img = cv2.resize(img, (w, h))
    img, info = findEncodedFace(img)
    pError = trackEncodedFace(img, info, w, pid, pError)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

