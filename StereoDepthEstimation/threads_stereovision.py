import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import undistort_images

# Mediapipe for face detection
import mediapipe as mp
import time
from threading import Thread


class VideoStreamWidget(object):
    def __init__(self, src):
        self.name = str(src)
        self.num = 0
        self.capture = cv2.VideoCapture(src)
        self.frame = None  # Initialize frame attribute
        self.status = None
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if status:
                    self.frame = frame  # Update frame attribute
                    self.status = status
            time.sleep(0.01)
    
    def show_frame(self):
        if self.frame is not None:  # Check if frame exists
            cv2.imshow(self.name, self.frame)
            return self.status
        return False
    
    def save_frame(self):
        if self.frame is not None:  # Check if frame exists
            if self.name == str(left_src):
                cv2.imwrite('images/stereoLeft/imageL' + str(self.num) + '.png', self.frame)
                print("Left image saved!")
            elif self.name == str(right_src):
                cv2.imwrite('images/stereoright/imageR' + str(self.num) + '.png', self.frame)
                print("Right image saved!")
            else:
                print("Error storing images")
            self.num += 1
        else:
            print("Error capturing frame")
    def get_frame(self):
        if self.frame is not None and self.name is not None:  # Check if frame exists
            return self.frame

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()


mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Open both cameras
left_src = "http://192.168.231.237:8080/video"
right_src = "http://192.168.231.251:8080/video"                 


# Stereo vision setup parameters
frame_rate = 10    #Camera frame rate (maximum at 120 fps)
B = 12               #Distance between the cameras [cm]
f = 26              #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]

left_widget = VideoStreamWidget(left_src)
right_widget = VideoStreamWidget(right_src)

# Main program loop with face detector and depth estimation using stereo vision
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    try:
        while True:
            succes_left= left_widget.show_frame()
            succes_right= right_widget.show_frame()
            frame_left= left_widget.get_frame()
            frame_right= right_widget.get_frame()


            ################## CALIBRATION #########################################################

            frame_right, frame_left = undistort_images.undistortRectify(frame_right, frame_left)

            ########################################################################################

            # If cannot catch any frame, break
            if not succes_right or not succes_left:                    
                break

            else:

                start = time.time()
                
                # Convert the BGR image to RGB
                frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
                frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

                # Process the image and find faces
                results_right = face_detection.process(frame_right)
                results_left = face_detection.process(frame_left)

                # Convert the RGB image to BGR
                frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
                frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


                ################## CALCULATING DEPTH #########################################################

                center_right = 0
                center_left = 0

                if results_right.detections:
                    for id, detection in enumerate(results_right.detections):
                        mp_draw.draw_detection(frame_right, detection)

                        bBox = detection.location_data.relative_bounding_box

                        h, w, c = frame_right.shape

                        boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                        center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                        cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

                if results_left.detections:
                    for id, detection in enumerate(results_left.detections):
                        mp_draw.draw_detection(frame_left, detection)

                        bBox = detection.location_data.relative_bounding_box

                        h, w, c = frame_left.shape

                        boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                        center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                        cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


                # If no ball can be caught in one camera show text "TRACKING LOST"
                if not results_right.detections or not results_left.detections:
                    cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                    cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

                else:
                    # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
                    # All formulas used to find depth is in video presentaion
                    depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

                    cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                    cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                    # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
                    print("Depth: ", str(round(depth,1)))


                end = time.time()
                totalTime = end - start

                fps = 1 / totalTime
                #print("FPS: ", fps)

                cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


                # Show the frames
                cv2.imshow("frame right", frame_right) 
                cv2.imshow("frame left", frame_left)


                key = cv2.waitKey(1)
                if key == ord('q'):
                    left_widget.release()
                    right_widget.release()
                    exit(1)
                elif key == ord('s'):
                    left_widget.save_frame()
                    right_widget.save_frame()
    except KeyboardInterrupt:
        left_widget.release()
        right_widget.release()

