import cv2

hem = cv2.VideoCapture("http://192.168.235.43:8080/video")
vet = cv2.VideoCapture("http://192.168.235.84:8080/video")
left_cam=hem
right_cam=vet
num = 0

while left_cam.isOpened() and right_cam.isOpened():

    succes1, left = left_cam.read()
    succes2, right = right_cam.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s') and succes1 and succes2: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', left)
        cv2.imwrite('images/stereoright/imageR' + str(num) + '.png', right)
        print("images saved!")
        num += 1
    cv2.imshow('left',left)
    cv2.imshow('right',right)