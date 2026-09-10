import cv2
import mediapipe as mp
import math
import time
import serial

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from geometry import calculate_angle


# ---------------- SERIAL COMMUNICATION ----------------

arduino = serial.Serial("COM4", 9600, timeout=1)
time.sleep(2)


# ---------------- HAND LANDMARK SETUP ----------------

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

fingers = {
    "Index": [5,6,7,8],
    "Middle": [9,10,11,12],
    "Ring": [13,14,15,16],
    "Pinky": [17,18,19,20]
}


# ---------------- HAND MODEL ----------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(options)


# ---------------- POSE MODEL ----------------

pose_base_options = python.BaseOptions(
    model_asset_path=r"C:\IOT self\pose_landmarker.task"
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_detector = vision.PoseLandmarker.create_from_options(pose_options)


# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

start_time = time.time()

finger_previous_states = {
    "Index": "STRAIGHT",
    "Middle": "STRAIGHT",
    "Ring": "STRAIGHT",
    "Pinky": "STRAIGHT"
}


# ---------------- SERIAL CONTROL VARIABLES ----------------

last_base_angle = None
last_elbow_angle = None
last_gripper_angle = None

last_command_time = 0

COMMAND_DELAY = 0.3
ANGLE_DEADZONE = 3

base_command_angle = 90
elbow_command_angle = 90
gripper_command_angle = 10


# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # ---------------------------------------------------
    # HAND DETECTION
    # ---------------------------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int((time.time() - start_time) * 1000)

    hand_result = hand_detector.detect_for_video(
        mp_image,
        timestamp
    )


    hand_state = "NO HAND"
    bent_count = 0

    finger_states = {
        "Index": "STRAIGHT",
        "Middle": "STRAIGHT",
        "Ring": "STRAIGHT",
        "Pinky": "STRAIGHT"
    }


    if hand_result.hand_landmarks:

        hand = hand_result.hand_landmarks[0]

        # ---------------- DRAW HAND LANDMARKS ----------------

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )


        # ---------------- DRAW HAND CONNECTIONS ----------------

        for connection in connections:

            x1 = int(hand[connection[0]].x * w)
            y1 = int(hand[connection[0]].y * h)

            x2 = int(hand[connection[1]].x * w)
            y2 = int(hand[connection[1]].y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # ---------------- FINGER ANGLE CALCULATION ----------------

        for name, points in fingers.items():

            p1 = hand[points[0]]
            p2 = hand[points[1]]
            p3 = hand[points[2]]

            angle1 = calculate_angle(
                (p1.x, p1.y),
                (p2.x, p2.y),
                (p3.x, p3.y)
            )


            # ---------------- FINGER STATE ----------------

            if angle1 < 140:

                finger_state = "BENT"

            elif angle1 > 160:

                finger_state = "STRAIGHT"

            else:

                finger_state = finger_previous_states[name]


            finger_previous_states[name] = finger_state

            finger_states[name] = finger_state


        bent_count = list(finger_states.values()).count("BENT")


        # ---------------- HAND STATE ----------------

        if bent_count >= 3:

            hand_state = "CLOSED"

        elif bent_count <= 1:

            hand_state = "OPEN"


        # ---------------- BASE ANGLE ----------------

        wrist_x = hand[0].x

        base_angle = 60 + (wrist_x - 0.2) * (120 - 60) / (0.8 - 0.2)

        base_angle = max(60, min(120, base_angle))

        base_command_angle = int(base_angle)


        # ---------------- GRIPPER ----------------

        if hand_state == "OPEN":

            gripper_command_angle = 10

        elif hand_state == "CLOSED":

            gripper_command_angle = 73


    # ---------------------------------------------------
    # POSE DETECTION
    # ---------------------------------------------------

    pose_result = pose_detector.detect_for_video(
        mp_image,
        timestamp
    )


    elbow_angle = 0
    elbow_state = "NO POSE"


    if pose_result.pose_world_landmarks:

        pose = pose_result.pose_world_landmarks[0]

        pose_screen = pose_result.pose_landmarks[0]


        right_shoulder = pose[11]
        right_elbow = pose[13]
        right_wrist = pose[15]


        # ---------------- ELBOW ANGLE ----------------

        elbow_angle = calculate_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y)
        )


        elbow_command_angle = int(
            max(0, min(180, elbow_angle))
        )


        # ---------------- ELBOW STATE ----------------

        if elbow_angle > 160:

            elbow_state = "STRAIGHT"

        elif elbow_angle > 100:

            elbow_state = "PARTIALLY BENT"

        else:

            elbow_state = "BENT"


        # ---------------- DRAW POSE ----------------

        shoulder_x = int(pose_screen[11].x * w)
        shoulder_y = int(pose_screen[11].y * h)

        elbow_x = int(pose_screen[13].x * w)
        elbow_y = int(pose_screen[13].y * h)

        wrist_x = int(pose_screen[15].x * w)
        wrist_y = int(pose_screen[15].y * h)


        cv2.circle(
            frame,
            (shoulder_x, shoulder_y),
            7,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (elbow_x, elbow_y),
            9,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (wrist_x, wrist_y),
            7,
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


    # ---------------------------------------------------
    # SERIAL COMMAND
    # ---------------------------------------------------

    current_time = time.time()

    base_changed = (
        last_base_angle is None
        or abs(base_command_angle - last_base_angle) >= ANGLE_DEADZONE
    )

    elbow_changed = (
        last_elbow_angle is None
        or abs(elbow_command_angle - last_elbow_angle) >= ANGLE_DEADZONE
    )

    gripper_changed = (
        last_gripper_angle is None
        or gripper_command_angle != last_gripper_angle
    )


    if (
        current_time - last_command_time >= COMMAND_DELAY
        and (base_changed or elbow_changed or gripper_changed)
    ):

        command = f"{base_command_angle},{elbow_command_angle},{gripper_command_angle}\n"

        arduino.write(command.encode())

        print("Sent:", command.strip())

        last_base_angle = base_command_angle
        last_elbow_angle = elbow_command_angle
        last_gripper_angle = gripper_command_angle

        last_command_time = current_time


    # ===================================================
    #              CAMERA UI / REPRESENTATION
    # ===================================================

    # ---------------- TOP TITLE ----------------

    cv2.rectangle(
        frame,
        (15, 15),
        (430, 65),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "VISION-BASED ROBOTIC ARM",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    # ===================================================
    # HAND INFORMATION PANEL
    # ===================================================

    cv2.rectangle(
        frame,
        (15, 80),
        (350, 330),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "HAND ANALYSIS",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # Hand state

    cv2.putText(
        frame,
        f"Gesture: {hand_state}",
        (30, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Bent Fingers: {bent_count}/4",
        (30, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ---------------- FINGER STATUS ----------------

    finger_y = 210

    for name, state in finger_states.items():

        symbol = "OK" if state == "STRAIGHT" else "BENT"

        text = f"{name:<7} : {symbol}"

        cv2.putText(
            frame,
            text,
            (30, finger_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        finger_y += 27


    # ===================================================
    # ELBOW INFORMATION PANEL
    # ===================================================

    cv2.rectangle(
        frame,
        (15, 350),
        (350, 485),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "ARM ANALYSIS",
        (30, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Elbow Angle: {elbow_angle:.1f} deg",
        (30, 415),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"State: {elbow_state}",
        (30, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ===================================================
    # ROBOT COMMAND PANEL
    # ===================================================

    cv2.rectangle(
        frame,
        (15, 505),
        (350, 665),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "ROBOT COMMAND",
        (30, 535),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"BASE       : {base_command_angle} deg",
        (30, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"ELBOW      : {elbow_command_angle} deg",
        (30, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    gripper_text = "OPEN" if gripper_command_angle == 10 else "CLOSED"


    cv2.putText(
        frame,
        f"GRIPPER    : {gripper_text}",
        (30, 630),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    # ===================================================
    # ANGLE VISUALIZATION AT ELBOW
    # ===================================================

    if pose_result.pose_world_landmarks:

        # Show the numerical angle directly beside the elbow

        cv2.putText(
            frame,
            f"{elbow_angle:.0f} deg",
            (elbow_x + 15, elbow_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ---------------- QUIT ----------------

    cv2.putText(
        frame,
        "Press Q to quit",
        (w - 190, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    cv2.imshow("Frame", frame)


    if cv2.waitKey(1) == ord('q'):

        break


# ---------------- CLEANUP ----------------

cap.release()

cv2.destroyAllWindows()

arduino.close()