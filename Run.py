from ultralytics import YOLO
import cv2
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load the sound
alarm_sound = pygame.mixer.Sound('alarm.mp3')

# Load the YOLO model
model = YOLO('best.pt')

# Function to play alarm sound
def play_alarm():
    alarm_sound.play()

# Perform object detection and handle the detection
def detect_and_alert():
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference
        results = model(frame, imgsz=640, conf=0.6)
        
        # Assuming results is a list
        for result in results:
            # Check if the result contains detections
            if hasattr(result, 'boxes'):
                # Iterate over the detections
                for box in result.boxes:
                    # Extract detection information
                    cls = box.cls[0]  # Class index
                    conf = box.conf[0]  # Confidence score
                    xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                    
                    # Convert class index to class name if needed
                    class_names = model.names  # List of class names
                    class_name = class_names[int(cls)]
                    
                    # Check for fire detection
                    if class_name == 'fire':  # Replace 'fire' with the actual class name in your model
                        play_alarm()
                        time.sleep(1)  # Prevents the alarm from playing too frequently
                        break  # Stop checking further once fire is detected
            
            # Display the frame with detections (if any)
            if hasattr(result, 'plot'):
                frame_with_detections = result.plot()  # Render detections on the frame
                cv2.imshow('YOLO Detection', frame_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the detection
detect_and_alert()
