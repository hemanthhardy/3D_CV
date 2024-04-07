from threading import Thread
import cv2, time
 
left="http://192.168.235.43:8080/video"
right="http://192.168.235.84:8080/video"
#hem = cv2.VideoCapture("http://192.168.235.43:8080/video")
#vet = cv2.VideoCapture("http://192.168.235.84:8080/video")

class VideoStreamWidget(object):
    def __init__(self, src):
        self.name=str(src)
        self.num=0
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)
    
    def show_frame(self):
        # Display frames in main program
        cv2.imshow(self.name, self.frame)
    def get_frames(self):
        return self.frame
    def key_action(self,key):
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
        elif key == ord('s'): # wait for 's' key to save and exit
            print(self.name,left,right)
            if self.name==left:
                cv2.imwrite('images/stereoLeft/imageL' + str(self.num) + '.png', self.frame)
                print("images saved!")

            elif self.name==right:
                cv2.imwrite('images/stereoright/imageR' + str(self.num) + '.png', self.frame)
                print("images saved!")
            else:
                print("Error storing images")
            self.num += 1

        
if __name__ == '__main__':
    leftw = VideoStreamWidget(left)
    rightw =  VideoStreamWidget(right)
    while True:
        try:
            leftw.show_frame()
            rightw.show_frame()

            key = cv2.waitKey(1)
            leftw.key_action(key)
            rightw.key_action(key)
        except AttributeError:
            pass