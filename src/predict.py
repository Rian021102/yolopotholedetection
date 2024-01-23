import cv2
from ultralytics import YOLO

# Replace the following line with the path to your local video file
video_path = "/Users/rianrachmanto/miniforge3/project/trial_ultrlytics/mixkit-potholes-in-a-rural-road-25208-medium.mp4"
cap = cv2.VideoCapture(video_path)

# Replace with the correct path to your YOLO model weights
model = YOLO('/Users/rianrachmanto/miniforge3/yolov8n_custom/train2/weights/best.pt')

fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

show_boxes = True

out = cv2.VideoWriter('/Users/rianrachmanto/miniforge3/project/trial_ultrlytics/src/output.mp4', fourcc, fps, (frame_width, frame_height), True)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        pothole_found = False
        results = model(frame, imgsz=320, stream=True, verbose=True)
        for result in results:
            for box in result.boxes.cpu().numpy():
                r = box.xyxy[0].astype(int)
                label = model.names[int(box.cls)]
                pothole_found = True
                if show_boxes:
                    cv2.rectangle(frame, r[:2], r[2:], (0, 0, 255), 2)  # Red box
                    text = f"{label}"
                    cv2.putText(frame, text, (r[0], r[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if pothole_found:
            out.write(frame)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
