import cv2
import time
import grpc
from ultralytics import YOLO

import detection_pb2
import detection_pb2_grpc

def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = detection_pb2_grpc.DetectionServiceStub(channel)

    print("Loading YOLOv8 model for tracking...")
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0
    print("Streaming to Go Web UI at http://localhost:8080. Press Ctrl+C in terminal to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, stream=True, conf=0.5, verbose=False)

        frame_data = detection_pb2.FrameData()
        frame_data.timestamp = int(time.time() * 1000)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                class_name = result.names[int(box.cls)]
                track_id = int(box.id[0]) if box.id is not None else -1

                det = frame_data.detections.add()
                det.class_name = class_name
                det.confidence = conf
                det.x_min, det.y_min, det.x_max, det.y_max = x1, y1, x2, y2
                det.track_id = track_id

                # Draw UI overlay on frame
                color = (0, 255, 0) if class_name == "person" else (255, 165, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name} #{track_id} ({conf:.2f})"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode EVERY frame so the Web UI gets a live MJPEG video stream
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if success:
            frame_data.image_data = buffer.tobytes()

        try:
            stub.SendFrame(frame_data)
        except grpc.RpcError:
            pass

    cap.release()

if __name__ == "__main__":
    main()