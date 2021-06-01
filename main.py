import cv2
import numpy as np
# for face then eye detection
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint( point1, point2):
    return int((point1.x + point2.x)/2), int((point1.y + point2.y)/2)

def line_length( point1, point2):
    return hypot(point1[0] - point2[0], point1[1] - point2[1])


font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[4]), facial_landmarks.part(eye_points[5]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 1)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 1)

    ver_line_length = line_length(center_top, center_bottom)
    hor_line_length = line_length(left_point, right_point)

    ratio = hor_line_length / ver_line_length
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
        # gaze detection
        eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                               (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                               (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                               (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                               (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                               (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        # cv2.polylines(frame, [eye_region], True, 255, 1)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 1)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        gray_eye = cv2.resize(gray_eye, None, fx=5, fy=5)
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold eye", threshold_eye)

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(threshold_eye, cv2.MORPH_OPEN, kernel)
        cv2.imshow("Opening", opening)

        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width/2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        ratio = left_side_white/(right_side_white+0.0001)
        return ratio

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        
        landmarks = predictor(gray, face)

        # detect blinking
        right_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        left_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        if left_eye_ratio > 6 and right_eye_ratio > 6:
            cv2.putText(frame, "BLINKING", (50,150), font, 7, (255, 0, 0), 7)

        right_eye_gaze_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        left_eye_gaze_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = int(right_eye_gaze_ratio+left_eye_gaze_ratio/2)

        if gaze_ratio <= 0.05:
            cv2.putText(frame, "RIGHT", (50, 100), font, 7, (0, 0, 255), 7)
        elif gaze_ratio > 1.95:
            cv2.putText(frame, "LEFT", (50, 100), font, 7, (0, 0, 255), 7)
        else:
            print(left_eye_gaze_ratio)
            print(right_eye_gaze_ratio)
            cv2.putText(frame, "CENTER", (50, 100), font, 7, (0, 0, 255), 7)
        


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()