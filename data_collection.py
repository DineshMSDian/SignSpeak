import os
import warnings
import time
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
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    model_complexity = 0
)

# A Small workaround for ignoring mediapipe warnings
time.sleep(0.5)

# Now clear the screen and show clean interface
os.system('cls' if os.name == 'nt' else 'clear')


# Data Collection starts!!

gesture_name = input("Enter Gesture Name: ").strip().lower()

# Create folder structure
os.makedirs(f"datasets/raw/{gesture_name}", exist_ok = True)

# Recording Variables
SEQUENCE_LENGTH = 60        # 2 secs @30fps
SAMPLES_TO_COLLECT = 50     # 50 samples each gesture
frame_buffer = []
frame_counter = 0
sample_counter = 0
is_recording = False

print(f"\nCollecting {SAMPLES_TO_COLLECT} samples for gesture: {gesture_name}")
print("Press 's' to start recording each sample\n")

# loop automatically stops after collecting 50 samples.
while sample_counter < SAMPLES_TO_COLLECT:

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
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
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
        # print("Total Frame Data Shape:", final_frame_data.shape)  # Comment-out this. can be used only for debugging purpose


        # RECORDING LOGIC STARTS HERE
        
        # If recording, save frames to buffer
        if is_recording:
            frame_buffer.append(final_frame_data)
            frame_counter += 1

            cv2.putText(frame, f"RECORDING: {frame_counter}/{SEQUENCE_LENGTH}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (frame_width - 30, 30), 10, (0, 0, 255), -1)

            # After collecting 60 frames, save them as a sequance
            if frame_counter >= SEQUENCE_LENGTH:
                sequence_array = np.array(frame_buffer)

                # Storing sample in the designated location
                file_name = f"sample_{sample_counter + 1:03d}.npy"
                file_path = f"datasets/raw/{gesture_name}/{file_name}"
                np.save(file_path, sequence_array)

                # Print confirmation
                sample_counter += 1
                print(f"✓ Saved sample {sample_counter}/{SAMPLES_TO_COLLECT}: {file_name}")

                # Reset for next frames
                frame_buffer = []
                frame_counter = 0
                is_recording = False

                if sample_counter < SAMPLES_TO_COLLECT:
                    print("  Press 's' to record next sample...")

        # Display status even when not recording
        cv2.putText(frame, f"Samples: {sample_counter}/{SAMPLES_TO_COLLECT}", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if not is_recording:
            cv2.putText(frame, "Press 's' to START", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # Write the frame to the output file
        # out.write(frame)
        # Display the captured frame
        cv2.imshow("Capture", frame)

        # prss 's' to enter recording , press 'q' to exit the loop
        key = cv2.waitKey(1) & 0xff
        
        if key == ord('s') and not is_recording:
            is_recording = True

        elif key == ord('q'):
            print("\Stopped by the user")
            break

    else:
        print("Frames not Captured")
        break

# release the capture and writer objects
camera.release()
# out.release()
cv2.destroyAllWindows()

# Dev notes for, data collection stats
print("\n" + "="*50)
print(f"Collection complete for '{gesture_name}'")
print(f"Total samples collected: {sample_counter}")
print(f"Saved to: dataset/raw/{gesture_name}/")
print("="*50 + "\n")