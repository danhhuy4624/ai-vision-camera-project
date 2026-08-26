package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net"

	"ai-camera/pb"

	"google.golang.org/grpc"

	// Import the pure-go sqlite driver anonymously
	_ "modernc.org/sqlite"
)

type server struct {
	pb.UnimplementedDetectionServiceServer
	db *sql.DB // Attach the database connection to the server struct
}

// initDB creates the SQLite file and builds the schema
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

// SendFrame receives the payload and persists it to SQLite
func (s *server) SendFrame(ctx context.Context, req *pb.FrameData) (*pb.Response, error) {
	if len(req.Detections) == 0 {
		return &pb.Response{Success: true, Message: "Empty frame ignored"}, nil
	}

	// Prepare the statement for secure, high-performance batch inserting
	stmt, err := s.db.Prepare(`
		INSERT INTO detections (timestamp, class_name, confidence, x_min, y_min, x_max, y_max) 
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`)
	if err != nil {
		log.Printf("Failed to prepare statement: %v", err)
		return &pb.Response{Success: false, Message: "Database error"}, err
	}
	defer stmt.Close()

	// Execute the prepared statement for each detection in the frame
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
	return &pb.Response{Success: true, Message: "Detections saved"}, nil
}

func main() {
	// 1. Initialize the SQLite Database
	db, err := initDB("./camera_logs.db")
	if err != nil {
		log.Fatalf("Database initialization failed: %v", err)
	}
	defer db.Close()

	// 2. Open TCP listener
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	// 3. Create and configure the gRPC server
	grpcServer := grpc.NewServer()

	// Pass the database connection into our server instance
	pb.RegisterDetectionServiceServer(grpcServer, &server{db: db})

	log.Printf("Go gRPC backend listening on %v", lis.Addr())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
