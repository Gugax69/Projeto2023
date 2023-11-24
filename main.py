import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch

def main():
    #args = parse_arguments()
    #frame_width, frame_height = args.webcam_resolution

    # Capture the webcam
    cap = cv2.VideoCapture(0)

    #Capture the example video (this was used for testing an example video)
    #cap = cv2.VideoCapture("D:\Downloads\yolov8-live-master(1)\Example videos\people_walking.mp4")

    #Path to save the output video (this was used for testing using an example video)
    #output_path = "D:\Downloads\yolov8-live-master(1)\Example videos\people_walking_output.mp4"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Load the YOLO model
    model = YOLO("yolov8l.pt")#.to("cuda")
    
    # Box annotator to draw the bounding boxes
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # #Video writer to save the output video (this was used for testing using an example video)
    # video_writer = cv2.VideoWriter(
    #     output_path,
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     cap.get(cv2.CAP_PROP_FPS),
    #     (frame_width, frame_height)
    # )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        # Detect only people (person is class 0 in COCO - Common Objects in Context)
        detections = detections[detections.class_id == 0]

        # Labels for the detected objects
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
        
        #passing_count = detections.count_passing(line_start, line_end)
        passing_count = len(detections)
        frame = cv2.putText(frame, f"People in frame: {passing_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # #Save the output video (this was used for testing using an example video)
        #video_writer.write(frame)

        # Show the frame (in this case the webcam feed)
        cv2.imshow("Yolov8 Webcam", frame)

        if (cv2.waitKey(30) == 27):
            break
  
    # Release the video writer (this was used for testing using an example video)
    #video_writer.release()

    cap.release()
    cv2.destroyAllWindows()  

if __name__ == "__main__":
    main()