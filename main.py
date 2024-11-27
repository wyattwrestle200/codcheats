from capture_screen import capture_screen
from process_frame import process_frame
from display_frame import display_frame
import pyautogui
import cv2

# Define screen region
region = {"top": 0, "left": 0, "width": pyautogui.size().width, "height": pyautogui.size().height}

while True:
    # Capture, process, and display
    frame = capture_screen(region)
    mask = process_frame(frame)
    display_frame(frame, mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
