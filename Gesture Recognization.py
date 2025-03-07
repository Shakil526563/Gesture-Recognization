import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

def recognize_gesture(landmarks):
    if landmarks is None:
        return None

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    fingers = [index_tip, middle_tip, ring_tip, pinky_tip]
    
    if thumb_tip.y < thumb_ip.y and all(f.y > middle_mcp.y + 0.02 for f in fingers):
        return "👍 Thumbs Up"
    
    if thumb_tip.y > thumb_ip.y and all(f.y > middle_mcp.y for f in fingers):
        return "👎 Thumbs Down"
    
    if all(f.y < middle_mcp.y - 0.02 for f in fingers) and thumb_tip.y < thumb_ip.y:
        return "✋ Open Hand"
    
    if index_tip.y < index_mcp.y - 0.02 and all(f.y > middle_mcp.y + 0.02 for f in fingers[1:]):
        return "☝️ Index Finger Up"
    
    if index_tip.y < index_mcp.y - 0.02 and middle_tip.y < middle_mcp.y - 0.02 and all(f.y > middle_mcp.y for f in fingers[2:]):
        return "✌️ Peace Sign"
    
    if index_tip.y < index_mcp.y - 0.02 and pinky_tip.y < middle_mcp.y - 0.02 and all(f.y > middle_mcp.y for f in fingers[1:3]):
        return "🤘 Rock Sign"
    
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks.landmark)
            if gesture:
                gesture_text = gesture

    cv2.putText(frame, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()