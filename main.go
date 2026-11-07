package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"ai-camera/pb"

	"google.golang.org/grpc"
	"gopkg.in/gomail.v2"
	_ "modernc.org/sqlite"
)

type server struct {
	pb.UnimplementedDetectionServiceServer
	db *sql.DB
}

var (
	notifiedIDs = make(map[int32]bool)
	emailMutex  sync.Mutex

	// Live video frame buffer for Web UI
	latestFrame []byte
	frameMutex  sync.RWMutex
)

type DetectionRecord struct {
	ID         int     `json:"id"`
	Timestamp  int64   `json:"timestamp"`
	TimeStr    string  `json:"time_str"`
	ClassName  string  `json:"class_name"`
	Confidence float32 `json:"confidence"`
	TrackID    int32   `json:"track_id"`
}

type DashboardStats struct {
	TotalDetections int               `json:"total_detections"`
	UniquePersons   int               `json:"unique_persons"`
	AlertsSent      int               `json:"alerts_sent"`
	RecentLogs      []DetectionRecord `json:"recent_logs"`
	Snapshots       []string          `json:"snapshots"`
}

func initDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, err
	}
	createTableQuery := `
	CREATE TABLE IF NOT EXISTS detections (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp INTEGER,
		class_name TEXT,
		confidence REAL,
		x_min REAL,
		y_min REAL,
		x_max REAL,
		y_max REAL,
		track_id INTEGER
	);`
	_, err = db.Exec(createTableQuery)
	return db, err
}

func sendEmailAlert(className string, imageBytes []byte, timestamp int64, trackID int32, personCount int) {
	t := time.UnixMilli(timestamp)
	filename := fmt.Sprintf("alert_%d.jpg", timestamp)

	if err := os.WriteFile(filename, imageBytes, 0644); err != nil {
		log.Printf("Failed to save image locally: %v", err)
		return
	}

	fromEmail := "darkcookieman88@gmail.com"
	toEmail := "nguyendanhhuy5@gmail.com"
	appPassword := "osja shla nlep absj"
	readableTime := t.Format("January 02, 2006 at 03:04:05 PM")

	var subject, body string
	if personCount > 1 {
		subject = fmt.Sprintf("Camera Alert: Another %s Detected (%d People in View)", className, personCount)
		body = fmt.Sprintf(
			"Your AI camera detected another %s appearance (Track ID #%d) on %s.\nThe current count of people on the webcam is %d. See the attached snapshot.",
			className, trackID, readableTime, personCount,
		)
	} else {
		subject = fmt.Sprintf("Camera Alert: %s (ID #%d) Detected", className, trackID)
		body = fmt.Sprintf(
			"Your AI camera detected a %s (Track ID #%d) on %s.\nThe current count of people on the webcam is 1. See the attached snapshot.",
			className, trackID, readableTime,
		)
	}

	m := gomail.NewMessage()
	m.SetHeader("From", fromEmail)
	m.SetHeader("To", toEmail)

	m.SetHeader("Subject", subject)
	m.SetBody("text/plain", body)

	m.Attach(filename)

	m.SetHeader("Subject", fmt.Sprintf("Camera Alert: %s (ID #%d) Detected", className, trackID))
	m.SetBody("text/plain", fmt.Sprintf("Your AI camera detected a %s (Track ID #%d) on %s.", className, trackID, readableTime))
	m.Attach(filename)

	d := gomail.NewDialer("smtp.gmail.com", 587, fromEmail, appPassword)
	if err := d.DialAndSend(m); err != nil {
		log.Printf("Failed to send email alert: %v", err)
		return
	}
	log.Printf("--> Email alert sent for ID #%d!", trackID)
}

func (s *server) SendFrame(ctx context.Context, req *pb.FrameData) (*pb.Response, error) {
	// 1. Update live video stream buffer for the Web UI
	if len(req.ImageData) > 0 {
		frameMutex.Lock()
		latestFrame = req.ImageData
		frameMutex.Unlock()
	}

	if len(req.Detections) == 0 {
		return &pb.Response{Success: true, Message: "Frame updated"}, nil
	}

	// 2. Insert detections into SQLite
	stmt, err := s.db.Prepare(`
		INSERT INTO detections (timestamp, class_name, confidence, x_min, y_min, x_max, y_max, track_id) 
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`)
	if err == nil {
		defer stmt.Close()
		for _, det := range req.Detections {
			_, _ = stmt.Exec(req.Timestamp, det.ClassName, det.Confidence, det.XMin, det.YMin, det.XMax, det.YMax, det.TrackId)
		}
	}

	// 3. Count how many people are currently visible in this frame
	currentPersonCount := 0
	for _, det := range req.Detections {
		if det.ClassName == "person" {
			currentPersonCount++
		}
	}

	// 4. Stateful ID Email Alert Trigger
	for _, det := range req.Detections {
		if det.ClassName == "person" && det.TrackId != -1 {
			emailMutex.Lock()
			if !notifiedIDs[det.TrackId] {
				notifiedIDs[det.TrackId] = true
				emailMutex.Unlock()

				if len(req.ImageData) > 0 {
					go sendEmailAlert(det.ClassName, req.ImageData, req.Timestamp, det.TrackId, currentPersonCount)
				}
				break
			}
			emailMutex.Unlock()
		}
	}

	return &pb.Response{Success: true, Message: "Processed"}, nil
}

// HTTP handler for live MJPEG video stream
func videoStreamHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	for {
		frameMutex.RLock()
		frame := latestFrame
		frameMutex.RUnlock()

		if len(frame) > 0 {
			fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", len(frame))
			if _, err := w.Write(frame); err != nil {
				break
			}
			fmt.Fprint(w, "\r\n")
		}
		time.Sleep(33 * time.Millisecond) // ~30 FPS
	}
}

// HTTP handler for dashboard statistics & activity logs
func statsHandler(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		stats := DashboardStats{RecentLogs: []DetectionRecord{}, Snapshots: []string{}}

		_ = db.QueryRow(`SELECT COUNT(*) FROM detections`).Scan(&stats.TotalDetections)
		_ = db.QueryRow(`SELECT COUNT(DISTINCT track_id) FROM detections WHERE class_name = 'person' AND track_id != -1`).Scan(&stats.UniquePersons)

		emailMutex.Lock()
		stats.AlertsSent = len(notifiedIDs)
		emailMutex.Unlock()

		rows, err := db.Query(`SELECT id, timestamp, class_name, confidence, track_id FROM detections ORDER BY id DESC LIMIT 25`)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var rec DetectionRecord
				if err := rows.Scan(&rec.ID, &rec.Timestamp, &rec.ClassName, &rec.Confidence, &rec.TrackID); err == nil {
					rec.TimeStr = time.UnixMilli(rec.Timestamp).Format("15:04:05")
					stats.RecentLogs = append(stats.RecentLogs, rec)
				}
			}
		}

		// List saved alert snapshots
		files, _ := filepath.Glob("alert_*.jpg")
		sort.Sort(sort.Reverse(sort.StringSlice(files)))
		if len(files) > 8 {
			files = files[:8]
		}
		stats.Snapshots = files

		json.NewEncoder(w).Encode(stats)
	}
}

func main() {
	db, err := initDB("./camera_logs.db")
	if err != nil {
		log.Fatalf("DB init failed: %v", err)
	}
	defer db.Close()

	// Start HTTP Web UI Server on :8080
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "index.html")
	})
	http.HandleFunc("/video", videoStreamHandler)
	http.HandleFunc("/api/stats", statsHandler(db))
	http.HandleFunc("/snapshots/", func(w http.ResponseWriter, r *http.Request) {
		file := strings.TrimPrefix(r.URL.Path, "/snapshots/")
		if strings.HasPrefix(file, "alert_") && strings.HasSuffix(file, ".jpg") {
			http.ServeFile(w, r, file)
		}
	})

	go func() {
		log.Println("Web UI Dashboard available at http://localhost:8080")
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// Start gRPC Server on :50051
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterDetectionServiceServer(grpcServer, &server{db: db})

	log.Printf("Go gRPC backend listening on %v", lis.Addr())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
