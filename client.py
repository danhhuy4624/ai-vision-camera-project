import cv2
import time
import grpc
from ultralytics import YOLO

import detection_pb2
import detection_pb2_grpc

def main():
    # Connect to the Go server
    channel = grpc.insecure_channel('localhost:50051')
    stub = detection_pb2_grpc.DetectionServiceStub(channel)

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

    print("Press 'q' in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True, conf=0.5)

        frame_data = detection_pb2.FrameData()
        frame_data.timestamp = int(time.time() * 1000)

        for result in results:
            for box in result.boxes:
                # Extract YOLO data
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                class_name = result.names[int(box.cls)]

                # Build the Protobuf payload
                det = frame_data.detections.add()
                det.class_name = class_name
                det.confidence = conf
                det.x_min = x1
                det.y_min = y1
                det.x_max = x2
                det.y_max = y2

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Fire the gRPC request if objects were detected
        if len(frame_data.detections) > 0:
            try:
                stub.SendFrame(frame_data)
            except grpc.RpcError:
                pass # Ignore errors if Go server is offline

        cv2.imshow("AI Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()