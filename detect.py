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

    print("Webcam stream started. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, stream=True, conf=0.5)

        frame_data = detection_pb2.FrameData()
        frame_data.timestamp = int(time.time() * 1000)
        person_detected = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                class_name = result.names[int(box.cls)]
                
                # NEW: Extract the unique tracking ID (default to -1 if lost)
                track_id = int(box.id[0]) if box.id is not None else -1

                if class_name == "person":
                    person_detected = True

                # Build Protobuf payload
                det = frame_data.detections.add()
                det.class_name = class_name
                det.confidence = conf
                det.x_min = x1
                det.y_min = y1
                det.x_max = x2
                det.y_max = y2
                det.track_id = track_id # Inject the ID

                # Draw bounding box with the unique ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name} ID:{track_id} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode image for alerts
        if person_detected:
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                frame_data.image_data = buffer.tobytes()

        # Stream to Go backend
        if len(frame_data.detections) > 0:
            try:
                stub.SendFrame(frame_data)
            except grpc.RpcError:
                pass 

        cv2.imshow("AI Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()