import cv2
import numpy as np
import mss
import pyautogui
import argparse
import Quartz.CoreGraphics as CG
import time


def get_game_window(window_name):
    """Finds the game window with the given title using Quartz."""
    # This function retrieves all window information
    windows = CG.CGEventCreate(None)
    windows_list = []
    # Add the logic to find the game window by title using Quartz API.
    # Mac doesn't directly support finding windows by title like in Windows.
    # Use tools like Accessibility API for deeper access to window info.
    raise NotImplementedError("Window finding logic for macOS is not trivial. Use Accessibility API.")


def detect_and_overlay(screen, model, conf_threshold, overlay_color, display_box):
    """Detects objects and overlays bounding boxes and labels."""
    frame = np.array(screen)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Get image dimensions
    height, width = frame.shape[:2]

    # Prepare the image for object detection
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

    # Get the game window coordinates
    try:
        game_window = get_game_window(args.window_name)
    except Exception as e:
        print(e)
        return

    # Assuming `game_window` gives a tuple of (left, top, width, height)
    monitor = {
        "top": game_window[1],
        "left": game_window[0],
        "width": game_window[2] - game_window[0],
        "height": game_window[3] - game_window[1],
    }

    with mss.mss() as sct:
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

    # Prompt the user before running the script
    start_script = input("Do you want to start the detection overlay? (y/n): ").strip().lower()
    if start_script == 'y':
        print("Starting detection overlay...")
        main(args)
    else:
        print("Exiting. No detection overlay will run.")
