import cv2
import face_recognition


cap = cv2.VideoCapture(0)
test = face_recognition.load_image_file("imgs/elon musk test.jpg")

if not cap.isOpened():
    print("Error: Cannot access the camera")
    exit()

if test is None:
    print("Error: Cannot load test image")
    exit()

test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_enc = face_recognition.face_encodings(test)[0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_loc = face_recognition.face_locations(rgb_frame)
    face_enc = face_recognition.face_encodings(rgb_frame, face_loc)

    color = (0, 0, 255)

    for i in range(0, len(face_loc)):
        (top, right, bottom, left) = face_loc[i]
        result = face_recognition.compare_faces([face_enc[i]], test_enc)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, "Elon Musk" if(result[0]) else "Unknow" , (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
