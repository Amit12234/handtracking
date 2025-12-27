import cv2 
import mediapipe as mp 
import time 

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
            self.mode = mode
            self.maxHands = maxHands
            self.detectionCon = detectionCon
            self.trackCon = trackCon

            self.mpHands = mp.solutions.hands
            self.mpDraw = mp.solutions.drawing_utils
            self.hands = self.mpHands.Hands(static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,)
            

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # Draw hand landmarks on the original BGR image.
        return img
    
    def findPosition(self, img, handNo=0, draw= True):
            lmList = []
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]  
                for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # print(id, cx, cy) # Print the landmark ID and its pixel coordinates.
                        lmList.append([id, cx, cy])
                        # if id == 4: # If the landmark is the tip of the thumb.
                        if draw:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # Draw a circle at the thumb tip
            return lmList
        

    

def main():

    cap = cv2.VideoCapture(0) # 0 is usually the default camera not 1. 
    pTime = 0
    cTime = 0 # Variables for calculating FPS (Frames Per Second)
    detector = handDetector() # Create an instance of the handDetector class.
    while True:
        success, img = cap.read()
        img = detector.findHands(img) # Detect and draw hands on the image.
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB. only uses RGB format.
        # results = hands.process(imgRGB) # Process the RGB image to detect hands.
        #print(results.multi_hand_landmarks) # Print the hand landmarks if detected.
        lmList = detector.findPosition(img) # Get the list of hand landmark positions.
        if len(lmList) != 0:
            print(lmList[4]) # Print the position of landmark with ID 4 (tip of the thumb). 

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        



if __name__ == "__main__":
    main()