import cv2

import mediapipe as mp         #we used this for the pretrained model

import time                    #we used this to like timestamp every frame

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


PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


pose_base_options = BaseOptions(

    model_asset_path=r"C:\IOT self\pose_landmarker.task"

)


pose_options = PoseLandmarkerOptions(

    base_options=pose_base_options,

    running_mode=VisionRunningMode.VIDEO,

    num_poses=1

)


pose_detector = PoseLandmarker.create_from_options(pose_options)


connections = [          #these are used to create lines that connect the dots in the live video feed

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


fingers = {          #this is used to get the angles of the fingers.

    "Index": [5, 6, 7, 8],

    "Middle": [9, 10, 11, 12],

    "Ring": [13, 14, 15, 16],

    "Pinky": [17, 18, 19, 20]

}


finger_previous_states = {

    "Index": "STRAIGHT",

    "Middle": "STRAIGHT",

    "Ring": "STRAIGHT",

    "Pinky": "STRAIGHT"

}


hand_state = "OPEN"


while camera.isOpened():         #this loop will run while the cam is open or until we press 'q' on the keyboard

    ret, frame = camera.read()

    if not ret:

        print("Can't receive frame. Exiting...")

        break


    frame = cv2.flip(frame, 1)


    height = frame.shape[0]

    width = frame.shape[1]


    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_image

    )


    timestamp = int(time.time() * 1000)             #second to miliseconds


    result = detector.detect_for_video(

        mp_image,

        timestamp

    )


    pose_result = pose_detector.detect_for_video(           #this is used to get the pose(for this project we only need the position of elbow and shoulder)

        mp_image,

        timestamp

    )


    if pose_result.pose_world_landmarks:

        pose = pose_result.pose_world_landmarks[0]

        pose_screen = pose_result.pose_landmarks[0]


        right_shoulder = pose[11]

        right_elbow = pose[13]

        right_wrist = pose[15]


        elbow_angle = calculate_angle(

            (

                right_shoulder.x,

                right_shoulder.y,

                right_shoulder.z

            ),

            (

                right_elbow.x,

                right_elbow.y,

                right_elbow.z

            ),

            (

                right_wrist.x,

                right_wrist.y,

                right_wrist.z

            )

        )


        cv2.putText(

            frame,

            f"Elbow: {elbow_angle:.1f}",

            (20, 160),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        print("Elbow:", elbow_angle)


        if elbow_angle > 150:

            elbow_state = "STRAIGHT"

        elif elbow_angle > 75:

            elbow_state = "PARTIALLY BENT"

        else:

            elbow_state = "BENT"


        cv2.putText(

            frame,

            f"Elbow: {elbow_state}",

            (20, 200),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        shoulder_x = int(pose_screen[11].x * width)

        shoulder_y = int(pose_screen[11].y * height)


        elbow_x = int(pose_screen[13].x * width)

        elbow_y = int(pose_screen[13].y * height)


        wrist_x = int(pose_screen[15].x * width)

        wrist_y = int(pose_screen[15].y * height)


        cv2.circle(

            frame,

            (shoulder_x, shoulder_y),

            8,

            (255, 0, 0),

            -1

        )


        cv2.circle(

            frame,

            (elbow_x, elbow_y),

            8,

            (255, 0, 0),

            -1

        )


        cv2.circle(

            frame,

            (wrist_x, wrist_y),

            8,

            (255, 0, 0),

            -1

        )


        cv2.line(

            frame,

            (shoulder_x, shoulder_y),

            (elbow_x, elbow_y),

            (255, 0, 0),

            3

        )


        cv2.line(

            frame,

            (elbow_x, elbow_y),

            (wrist_x, wrist_y),

            (255, 0, 0),

            3

        )


    if result.hand_landmarks:       #it will run the following if a hand is detected

        hand = result.hand_landmarks[0]


        wrist_x = hand[0].x


        base_angle = 60 + (wrist_x - 0.2) * (120 - 60) / (0.8 - 0.2)        #this is used to get the angle of the wrist, to prevent any issues with the robot we have decided to clamp it to between 60 and 120 degrees


        base_angle = max(60, min(120, base_angle))


        p5 = (hand[5].x, hand[5].y)

        p6 = (hand[6].x, hand[6].y)

        p7 = (hand[7].x, hand[7].y)

        p8 = (hand[8].x, hand[8].y)


        finger_states = []


        for name, landmarks in fingers.items():

            angle1 = calculate_angle(

                (hand[landmarks[0]].x, hand[landmarks[0]].y),

                (hand[landmarks[1]].x, hand[landmarks[1]].y),

                (hand[landmarks[2]].x, hand[landmarks[2]].y)

            )


            angle2 = calculate_angle(

                (hand[landmarks[1]].x, hand[landmarks[1]].y),

                (hand[landmarks[2]].x, hand[landmarks[2]].y),

                (hand[landmarks[3]].x, hand[landmarks[3]].y)

            )


            if angle1 < 140:

                finger_state = "BENT"

                finger_previous_states[name] = finger_state

                finger_states.append(finger_state)


            elif angle1 > 160:

                finger_state = "STRAIGHT"

                finger_previous_states[name] = finger_state

                finger_states.append(finger_state)


            else:

                finger_state = finger_previous_states[name]

                finger_previous_states[name] = finger_state

                finger_states.append(finger_state)


            print(name, angle1, angle2, finger_state)   #this will print the name of the finger, the angles and the state of the finger in the terminal,also updated to fingger_status being shown in the terminal.


        bent_count = finger_states.count("BENT")


        if bent_count >= 3:

            hand_state = "CLOSED"

        elif bent_count <= 1:

            hand_state = "OPEN"


        cv2.putText(

            frame,

            f"Hand: {hand_state}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Bent fingers: {bent_count}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Base: {base_angle:.1f}",

            (20, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


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

                (y1, y2),

                (0, 255, 0),

                2

            )


    cv2.imshow("Frame", frame)


    if cv2.waitKey(1) == ord('q'):        #press 'q' to exit the loop

        break


camera.release()

cv2.destroyAllWindows()        #destroy/close all the cam windows