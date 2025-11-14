import cv2
import face_recognition


img = face_recognition.load_image_file("imgs/elon musk.jpg")
test = face_recognition.load_image_file("imgs/elon musk test.jpg")

if img is None or test is None:
    print("Error: images not found!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)


img_faceLoc = face_recognition.face_locations(img)[0]
img_faceEnc = face_recognition.face_encodings(img)[0]

test_faceLoc = face_recognition.face_locations(test)[0]
test_faceEnc = face_recognition.face_encodings(test)[0]

cv2.rectangle(img, (img_faceLoc[3], img_faceLoc[0]), (img_faceLoc[1], img_faceLoc[2]), (255, 0, 255), 2)
cv2.rectangle(test, (test_faceLoc[3], test_faceLoc[0]), (test_faceLoc[1], test_faceLoc[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([img_faceEnc], test_faceEnc)
face_dis = face_recognition.face_distance([img_faceEnc], test_faceEnc)

print(result, face_dis)

cv2.imshow("image", img)
cv2.imshow("test", test)


cv2.waitKey(0)
cv2.destroyAllWindows()
