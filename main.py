import cv2 

# Open the default camera
camera = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f"{frame_width} x {frame_height}")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('first_capture.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:

    # ret = return true if the frame available
    ret, frame = camera.read() 
    if ret == True:

        # Write the frame to the output file
        out.write(frame)

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
out.release()
cv2.destroyAllWindows()