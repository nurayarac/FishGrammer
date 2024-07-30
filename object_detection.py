import cv2
from ultralytics import YOLO

model = YOLO('C:\\Users\\nuray\\Untitled Folder\\best_nuray.pt')
video_path = 'C:\\Users\\nuray\\Untitled Folder\\deneme2.mp4'
cap = cv2.VideoCapture(video_path)

total_fish_count = 0
fps = 0

identified_fish_ids = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    start_time = cv2.getTickCount()
    results = model.track(frame, persist=True)
    frame_fish_count = 0
    for result in results:
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            for box, track_id in zip(result.boxes, track_ids):
                print(track_id)
                xyxy = box.xyxy[0].tolist()

                x1, y1, x2, y2 = map(int, xyxy)
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                label = f'{model.names[int(cls)]} {conf:.2f}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Sarı çerçeve
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Sarı etiket

                if model.names[int(cls)] == 'Fish - v9 2024-07-23 8-54am':
                    if track_id not in identified_fish_ids:
                        identified_fish_ids.append(track_id)
                        frame_fish_count += 1

    total_fish_count += frame_fish_count
    end_time = cv2.getTickCount()
    time_taken = (end_time - start_time) / cv2.getTickFrequency()
    fps = 1 / time_taken

    cv2.putText(frame, f'Total Fish Count: {total_fish_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
