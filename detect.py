import cv2
import time
import grpc
from ultralytics import YOLO

# Import the generated protobuf modules
import detection_pb2
import detection_pb2_grpc

def main():
    # 1. Connect to the Go gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = detection_pb2_grpc.DetectionServiceStub(channel)

    # 2. Load the YOLO model
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    
    # 3. Initialize webcam (DirectShow for Windows fast startup)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Webcam stream started. Press 'q' in the video window to stop.")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Run YOLO inference
        results = model(frame, stream=True, conf=0.5)

        # 5. Prepare the gRPC payload
        frame_data = detection_pb2.FrameData()
        frame_data.timestamp = int(time.time() * 1000)

        for result in results:
            for box in result.boxes:
                # Extract YOLO data
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                class_name = result.names[int(box.cls)]

                # Build the Protobuf payload for the Go server
                det = frame_data.detections.add()
                det.class_name = class_name
                det.confidence = conf
                det.x_min = x1
                det.y_min = y1
                det.x_max = x2
                det.y_max = y2

                # Draw bounding box and label on the local video frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 6. Fire the gRPC request if objects were detected
        if len(frame_data.detections) > 0:
            try:
                stub.SendFrame(frame_data)
            except grpc.RpcError:
                pass # Fail silently if the Go server is offline, keeps the video running

        # 7. Display the video frame
        cv2.imshow("AI Camera", frame)
        
        # Exit when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()