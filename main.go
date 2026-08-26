package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net"
	"os"
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

// Global variables for throttling email notifications safely across concurrent gRPC calls
var (
	lastEmailTime time.Time
	emailMutex    sync.Mutex
)

// initDB initializes the SQLite database file and creates the detections schema
func initDB(filepath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", filepath)
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
		y_max REAL
	);`

	_, err = db.Exec(createTableQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to create table: %w", err)
	}

	log.Println("SQLite database initialized successfully.")
	return db, nil
}

// sendEmailAlert writes the frame image to disk and delivers an email notification with attachment
func sendEmailAlert(className string, imageBytes []byte, timestamp int64) {
	// 1. Save frame locally as a JPEG
	filename := fmt.Sprintf("alert_%d.jpg", timestamp)
	err := os.WriteFile(filename, imageBytes, 0644)
	if err != nil {
		log.Printf("Failed to save image locally: %v", err)
		return
	}

	// 2. Configure credentials and construct message
	fromEmail := "darkcookieman88@gmail.com"
	toEmail := "nguyendanhhuy5@gmail.com"
	appPassword := "osja shla nlep absj"

	m := gomail.NewMessage()
	m.SetHeader("From", fromEmail)
	m.SetHeader("To", toEmail)
	m.SetHeader("Subject", fmt.Sprintf("Camera Alert: %s Detected", className))
	m.SetBody("text/plain", fmt.Sprintf("Your AI camera detected %s at timestamp %d. See the attached snapshot.", className, timestamp))
	m.Attach(filename)

	// 3. Dial Gmail SMTP on port 587
	d := gomail.NewDialer("smtp.gmail.com", 587, fromEmail, appPassword)

	if err := d.DialAndSend(m); err != nil {
		log.Printf("Failed to send email alert: %v", err)
		return
	}

	log.Println("--> Real-time email alert with photo sent successfully!")
}

// SendFrame handles incoming gRPC frame detections from the Python client
func (s *server) SendFrame(ctx context.Context, req *pb.FrameData) (*pb.Response, error) {
	if len(req.Detections) == 0 {
		return &pb.Response{Success: true, Message: "Empty frame ignored"}, nil
	}

	// 1. Insert detections into SQLite via prepared statement
	stmt, err := s.db.Prepare(`
		INSERT INTO detections (timestamp, class_name, confidence, x_min, y_min, x_max, y_max) 
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`)
	if err != nil {
		log.Printf("Failed to prepare statement: %v", err)
		return &pb.Response{Success: false, Message: "Database error"}, err
	}
	defer stmt.Close()

	for _, det := range req.Detections {
		_, err := stmt.Exec(
			req.Timestamp,
			det.ClassName,
			det.Confidence,
			det.XMin,
			det.YMin,
			det.XMax,
			det.YMax,
		)
		if err != nil {
			log.Printf("Failed to insert record: %v", err)
		}
	}
	fmt.Printf("Saved %d detections to SQLite at timestamp %d\n", len(req.Detections), req.Timestamp)

	// 2. Evaluate alert trigger (1-minute cooldown per notification)
	for _, det := range req.Detections {
		if det.ClassName == "person" {
			emailMutex.Lock()
			if time.Since(lastEmailTime) > time.Minute {
				lastEmailTime = time.Now()
				emailMutex.Unlock()

				// DEBUG: Check if image data exists
				if len(req.ImageData) > 0 {
					log.Println("Person detected and image received. Triggering email...")
					go sendEmailAlert(det.ClassName, req.ImageData, req.Timestamp)
				} else {
					log.Println("WARNING: Person detected, but Python did not send any image data!")
				}
				break
			}
			emailMutex.Unlock()
		}
	}

	return &pb.Response{Success: true, Message: "Detections processed"}, nil
}

func main() {
	// 1. Initialize SQLite Database
	db, err := initDB("./camera_logs.db")
	if err != nil {
		log.Fatalf("Database initialization failed: %v", err)
	}
	defer db.Close()

	// 2. Open TCP listener on port 50051
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	// 3. Start gRPC Server
	grpcServer := grpc.NewServer()
	pb.RegisterDetectionServiceServer(grpcServer, &server{db: db})

	log.Printf("Go gRPC backend listening on %v", lis.Addr())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
