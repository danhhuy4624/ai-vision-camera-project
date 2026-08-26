import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Load the pre-trained YOLOv8 Nano model (downloads automatically on first run)
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    # 2. Initialize the webcam using DirectShow for fast startup on Windows
    # If 0 does not open your camera, try changing it to 1 or 2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set camera resolution (optional: 640x480 is optimal for real-time CPU performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Webcam stream started. Press 'q' in the video window to exit.")

    prev_time = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 3. Run YOLO inference on the current frame
        results = model(frame, stream=True, conf=0.5)

        # 4. Extract the raw data manually
        for result in results:
            # result.boxes contains all the detections for the current frame
            for box in result.boxes:
                
                # A. Extract Coordinates (x1, y1, x2, y2)
                # .xyxy[0] gets the top-left and bottom-right pixel coordinates
                # .tolist() converts the PyTorch tensor into a standard Python list
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # B. Extract Confidence Score
                # .conf contains the confidence percentage (e.g., 0.85)
                # float() converts it to a standard Python float
                confidence = float(box.conf)

                # C. Extract Class Labels
                # .cls contains the numeric ID of the class (e.g., 0 for person)
                class_id = int(box.cls)
                
                # result.names is a dictionary mapping IDs to string names
                class_name = result.names[class_id]

                # Print the data to the terminal
                print(f"Detected {class_name} ({confidence*100:.1f}%) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                
                # Example custom logic: Only draw boxes for people
                if class_name == "person":
                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                     cv2.putText(frame, f"Intruder! {confidence:.2f}", (int(x1), int(y1)-10), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Use the manually modified frame for display instead of result.plot()
            annotated_frame = frame
            
        # 5. Display the annotated frame in a window
        cv2.imshow("AI Detection Camera Feed", annotated_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()