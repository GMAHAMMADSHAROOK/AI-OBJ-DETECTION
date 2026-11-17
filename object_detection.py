
import cv2
from ultralytics import YOLO
from collections import Counter

# Load AI model
model = YOLO("yolov8n.pt")

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working!")
        break

    # Detect objects
    results = model(frame, conf=0.60, verbose=False)

    # Draw detection boxes
    output = results[0].plot()

    # Count objects
    counts = Counter()

    if results[0].boxes is not None:
        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]
            percentage = confidence * 100

            counts[object_name] += 1

            print(
                object_name,
                "Confidence:",
                round(percentage, 2),
                "%"
            )

    # Display object count
    y = 30

    cv2.putText(
        output,
        "OBJECT COUNT",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    for name, count in counts.items():
        y += 30

        text = name + ": " + str(count)

        cv2.putText(
            output,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display camera
    cv2.imshow("AI Object Detection System", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
