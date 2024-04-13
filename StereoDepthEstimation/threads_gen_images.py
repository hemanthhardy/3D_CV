from threading import Thread
import cv2
import time

# Define the sources for left and right cameras
left_src = "http://192.168.231.237:8080/video"
right_src = "http://192.168.231.251:8080/video"

class VideoStreamWidget(object):
    def __init__(self, src):
        self.name = str(src)
        self.num = 0
        self.capture = cv2.VideoCapture(src)
        self.frame = None  # Initialize frame attribute
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if status:
                    self.frame = frame  # Update frame attribute
            time.sleep(0.01)
    
    def show_frame(self):
        if self.frame is not None:  # Check if frame exists
            cv2.imshow(self.name, self.frame)
        
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

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Create instances of VideoStreamWidget for left and right cameras
    left_widget = VideoStreamWidget(left_src)
    right_widget = VideoStreamWidget(right_src)

    try:
        while True:
            left_widget.show_frame()
            right_widget.show_frame()

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