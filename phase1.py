import cv2                  
import mediapipe as mp      #we used this for the pretrained model
import time                 #we used this to like timestamp every frame
from geometry import calculate_angle
camera = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,       #we also could have used LIVE_STREAM
    num_hands=1                                 #only 1 hand for now , can be updated 
)

detector = HandLandmarker.create_from_options(options)

connections =[          #these are used to create lines that connect the dots in the live video feed
    (0,1),
    (1,2),
    (2,3),
    (3,4),

    (0,5),
    (5,6),
    (6,7),
    (7,8),

    (0,9),
    (9,10),
    (10,11),
    (11,12),
    (0,13),
    (13,14),
    (14,15),
    (15,16),
    (0,17),
    (17,18),
    (18,19),
    (19,20),
    (5, 9),
(9, 13),
(13, 17)
]

while camera.isOpened():            #this loop will run while the cam is open or until we press 'q' on the keyboard

    ret, frame = camera.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    timestamp = int(time.time() * 1000)         #second to miliseconds 

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    if result.hand_landmarks:       #it will run the following if a hand is detected 
        hand = result.hand_landmarks[0]

        height = frame.shape[0]
        width = frame.shape[1]
        p6 = (hand[6].x, hand[6].y)
        p7 = (hand[7].x, hand[7].y)
        p8 = (hand[8].x, hand[8].y)
        index_angle = calculate_angle(p6, p7, p8)

        print(index_angle)
       
        for landmark in hand:

            landmark_x = int(landmark.x * width)    #these are the coordinates of the points 
            landmark_y = int(landmark.y * height)

            cv2.circle(
                frame,
                (landmark_x, landmark_y),
                6,
                (0, 0, 255),
                -1
            )
        for start, end in connections:

            point1 = hand[start]
            point2 = hand[end]

            x1 = int(point1.x * width)
            y1 = int(point1.y * height)

            x2 = int(point2.x * width)
            y2 = int(point2.y * height)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()     #destroy/close all the cam windows