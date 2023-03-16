import cv2
import mediapipe as mp
import os
import numpy as np
import json

# Read parameters ###########################################################
with open("parameters.json") as data:
    parameters=json.loads(data.read())


# Initial conditions ########################################################
#state='s'
state='r'
state=state.upper()
folder='resources/files/validation/'+state
# folder='resources/files/training/'+state
quantity=parameters['quantity']
width, height=parameters['width'], parameters['height']


# Camera ####################################################################
cap=cv2.VideoCapture(1)


# Create folder if it doesn't exists ########################################
if not os.path.exists(folder):
    os.makedirs(folder)


# Mediapipe #################################################################
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils


#Loop body #################################################################
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:
    cont=1
    capture=False
    while True:
        # Read image from camera
        ret, frame=cap.read()
        if ret==False:
            break

        # If ESC key pressed or get quantity imagen, then exit
        if cv2.waitKey(1) & 0xFF==27 or cont>quantity:
            break

        # If SPACE BAR key pressed, then capture and save image
        if cv2.waitKey(1) & 0xFF==32:
            capture=True

        # Getting backgroun image
        background=cv2.imread('resources/images/background.png')

        # Working mediapipe
        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=hands.process(frame_rgb)
        hands_image=results.multi_hand_landmarks
        if hands_image is not None:
            for hand_landmarks in hands_image:
                # Paintng dots and linies
                mp_drawing.draw_landmarks(
                    background, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    # Changing color
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=0, circle_radius=0),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=8)
                )

                # Fill triangle
                h, w, _= background.shape
                p1x=int((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x)*w)
                p1y=int((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)*h)
                p2x=int((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x)*w)
                p2y=int((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y)*h)
                p3x=int((hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x)*w)
                p3y=int((hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y)*h)
                p4x=int((hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x)*w)
                p4y=int((hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y)*h)
                p5x=int((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x)*w)
                p5y=int((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)*h)
                points = np.array([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y], [p5x, p5y]])
                cv2.fillPoly(background, pts=[points], color=(255, 255, 255))


        # Points for cut and rectangle
        x1, y1=220, 140 
        x2, y2=420, 340

        # Saving cut image
        if capture:
            cut=background[y1:y2, x1:x2]
            background=cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            cut=cv2.resize(cut, (height,width))
            cv2.imwrite(folder+'/'+state+'_{}.jpg'.format(cont), cut)
            cont+=1
            
        # Inverse horizontally image before show it
        background= cv2.flip(background, 1)

        # Put rectangle and text
        cv2.rectangle(img=background, pt1=(x1,y1), pt2=(x2,y2), color=(255, 255, 255), thickness=3)
        cv2.putText(background, state, (x1, y1 - 5), 1, 3, (255, 255, 255), 2, cv2.LINE_AA)
        if capture:
            cv2.putText(background, str(cont-1), (x2, y2 + 30), 1, 2, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Showing image
        cv2.imshow("Sign capture", background)
        

cap.release()
cv2.destroyAllWindows()