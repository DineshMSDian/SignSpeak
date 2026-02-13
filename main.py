import cv2 
import mediapipe as mp
import numpy as np

# Open the default camera
camera = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print(f"{frame_width} x {frame_height}")

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('first_capture.mp4', fourcc, 20.0, (frame_width, frame_height))

# Initializing Mediapipe

mp_hands = mp.solutions.hands                       # load ML model to track hands
mp_drawing = mp.solutions.drawing_utils             # Visualization Helper Tool
mp_drawing_styles = mp.solutions.drawing_styles     # (optional) for colour style

# Blueprint for the model
hand = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

while True:

    # ret = return true if the frame available
    ret, frame = camera.read() 
    if ret == True:

        # Convert BGR -> RGB frames since mediapipe only supports RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip the video frame, by default mediapipe acts like a mirror.
        frame = cv2.flip(frame, 1)
        
        # Process the RGB image
        result = hand.process(frame)

        #Convert RGB -> BGR after processing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # To store each hand's coordinates seperately
        left_hand_data = np.zeros(63)
        right_hand_data = np.zeros(63)

        # If hands are present in video (frame)
        if result.multi_hand_landmarks:
            # for hand_land_marks in result.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(frame, hand_land_marks, mp_hands.HAND_CONNECTIONS)
                
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style()
                )
                
                label = handedness.classification[0].label
                # Extract the math coordinates of left hand
                if label == 'Left':
                    temp_list_left = []
                    for points in hand_landmarks.landmark:
                        temp_list_left.append(points.x)
                        temp_list_left.append(points.y)
                        temp_list_left.append(points.z)

                    # Overwrite the zeros with extrcted datas
                    left_hand_data = np.array(temp_list_left)
                
                # Extract the math coordinates of right hand
                elif label == 'Right':
                    temp_list_right = []
                    for points in hand_landmarks.landmark:
                        temp_list_right.append(points.x)
                        temp_list_right.append(points.y)
                        temp_list_right.append(points.z)
                    
                    # Overwrite the zeros with extrcted datas
                    right_hand_data = np.array(temp_list_right)

        # Combine them into one flat 126-number array       
        final_frame_data = np.concatenate([left_hand_data, right_hand_data])
        print("Total Frame Data Shape:", final_frame_data.shape)

        # Write the frame to the output file
        # out.write(frame)

        # Display the captured frame
        cv2.imshow("Capture", frame)

        # press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        print("Frames not Captured")
        break

# release the capture and writer objects
camera.release()
# out.release()
cv2.destroyAllWindows()