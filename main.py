import cv2
import numpy as np
import mss
import pygetwindow as gw
import argparse

def detect_and_overlay(screen, model, conf_threshold, overlay_color, display_box):
    # Convert the screen capture to a format OpenCV can use
    frame = np.array(screen)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Get image dimensions
    height, width = frame.shape[:2]

    # Prepare blob for object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)

    # Run object detection
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    outputs = model.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box if enabled
                if display_box:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), overlay_color, 2)
                cv2.putText(frame, f"Enemy: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_color, 2)

    return frame

def main(args):
    # Load the pre-trained model (YOLO in this case)
    model = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    overlay_color = tuple(map(int, args.color.split(',')))
    conf_threshold = args.confidence

    with mss.mss() as sct:
        # Identify the window or screen to capture
        game_window = gw.getWindowsWithTitle(args.window_name)[0]
        monitor = {"top": game_window.top, "left": game_window.left, "width": game_window.width, "height": game_window.height}

        while True:
            # Capture the screen
            screen = sct.grab(monitor)
            frame = detect_and_overlay(screen, model, conf_threshold, overlay_color, args.box)

            # Display the results
            cv2.imshow("Detection Overlay", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enemy Detection Overlay Script")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to YOLO model configuration file")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to YOLO model weights file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--color", type=str, default="0,255,0", help="Overlay color in BGR format (e.g., '0,255,0')")
    parser.add_argument("--box", action="store_true", help="Enable bounding boxes around detected objects")
    parser.add_argument("--window_name", type=str, default="Xbox Game Pass", help="Window title to capture")
    args = parser.parse_args()

    main(args)
    