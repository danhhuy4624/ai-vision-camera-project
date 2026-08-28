# AI Vision Camera Project
This system is a real-time object detection and alerting pipeline. It features a Python client that captures webcam frames, runs YOLOv8 AI inference locally, and uses gRPC to stream detection payloads to a high-performance Go backend, which handles SQLite database persistence and automated email alerts.

## Project Technology Stack
Programming Languages: Python (for computer vision and YOLO inference) and Go (for the high-performance backend, concurrency, and gRPC communication).
Core Frameworks & Libraries: OpenCV (opencv-python) for frame capture and manipulation, PyTorch as the underlying deep learning engine, and Ultralytics YOLOv8 for real-time object detection.
Communication & Storage: gRPC and Protocol Buffers (Protobuf) for structured streaming between Python and Go, SQLite for local database persistence, and gomail for automated email alerts.

## How the Detection Works

### Frame Capture: 
The Python client captures a live webcam feed using OpenCV (optimized with DirectShow on Windows).

### AI Inference: 
Each frame is processed by the YOLOv8 model to locate objects and evaluate confidence scores.

### Payload Streaming: 
When a target class like a "person" is identified, the client encodes the frame into JPEG bytes and packages the detection metadata (bounding box coordinates and timestamps) into a gRPC payload sent to the Go backend.

## Expected Result

### Local Display: 
A real-time OpenCV window displays the video feed with green bounding boxes, class labels, confidence scores, and an FPS counter.

### Backend Logs & Persistence: 
The Go server receives the stream via gRPC, prints the detection details to the console, and saves the records to an SQLite database.

### Automated Alerts: 
Upon detecting a person, the system triggers a background email notification containing an attached snapshot of the event.

## Running the Full Pipeline Test
### Step 1: Start the Go Backend (Terminal 1)
Start the database and gRPC server by running:
`go run main.go`

*Wait until you see the following output in the console:*
> `SQLite database initialized successfully.`
> `Go gRPC backend listening on [::]:50051`

### Step 2: Start the Python Camera (Terminal 2)
Start the camera client in 2nd terminal by running:
`python detect.py`

*Wait until the webcam video window pops up on your screen.*

### Step 3: Trigger the AI
1. Look at the webcam video window on your screen.
2. Step into the frame so the AI can clearly see your head and shoulders.
3. Wait for the green bounding box to appear with the word **"person"** above it.

### Step 4: Verify the Success Logs
**In the Python Terminal, you should instantly see:**
> `DEBUG: Attached image of size XXXXX bytes`

**In the Go Terminal, you should instantly see:**
> `Saved 1 detections to SQLite...`
> `Person detected and image received. Triggering email...`
> `--> Real-time email alert with photo sent successfully!`

The debug outputs should appear and the notification mail should be sent to the recipient
