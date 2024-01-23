import cv2
from ultralytics import YOLO

# Replace the following line with the path to your local video file
video_path = "/Users/rianrachmanto/miniforge3/project/potholes_detection_using_yolo/test_videos/mixkit-potholes-in-a-rural-road-25208-medium.mp4"
cap = cv2.VideoCapture(video_path)

# Replace with the correct path to your YOLO model weights
model = YOLO('/Users/rianrachmanto/miniforge3/project/potholes_detection_using_yolo/best.pt')

fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

show_boxes = True

out = cv2.VideoWriter('/Users/rianrachmanto/miniforge3/project/trial_ultrlytics/src/output_count.mp4', fourcc, fps, (frame_width, frame_height), True)

unique_potholes = set()  # Set to store unique potholes
prev_potholes = set()  # Set to store potholes from the previous frame

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        results = model(frame, imgsz=320, stream=True, verbose=True)
        potholes_in_frame = set()  # Set to store potholes detected in the current frame

        for result in results:
            for box in result.boxes.cpu().numpy():
                r = box.xyxy[0].astype(int)
                label = model.names[int(box.cls)]
                pothole_key = f"{label}_{r[0]}_{r[1]}_{r[2]}_{r[3]}"  # Create a unique key for each pothole
                potholes_in_frame.add(pothole_key)

                if show_boxes:
                    cv2.rectangle(frame, r[:2], r[2:], (0, 0, 255), 2)  # Red box

        new_potholes = potholes_in_frame - prev_potholes
        unique_potholes.update(new_potholes)  # Add new potholes to the set

        if len(new_potholes) > 0:
            out.write(frame)

        prev_potholes = potholes_in_frame

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()

print(f"Total unique potholes detected: {len(unique_potholes)}")

cv2.destroyAllWindows()
