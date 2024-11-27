import cv2
import numpy as np
import tkinter as tk
from tkinter import StringVar
import pyautogui
import mss
import threading

# Initialize global variables
selected_color = None
running = False

# Define HSV color ranges
color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Blue": ([94, 80, 2], [126, 255, 255]),
    "Green": ([40, 40, 40], [70, 255, 255])
}

# Screen capture function
def capture_screen(region):
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(region))
        return screenshot

# Frame processing function
def process_frame(frame, lower, upper):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
    return mask

# Display function
def display_frame(frame, mask):
    highlighted = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Game Feed", frame)
    cv2.imshow("Highlighted Enemies", highlighted)

# Main detection loop
def detection_loop():
    global running
    global selected_color

    # Get screen size and define capture region
    region = {"top": 0, "left": 0, "width": pyautogui.size().width, "height": pyautogui.size().height}

    # Retrieve HSV range for selected color
    lower, upper = color_ranges[selected_color]

    while running:
        frame = capture_screen(region)
        mask = process_frame(frame, lower, upper)
        display_frame(frame, mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()

# Start button callback
def start_detection():
    global running
    global selected_color

    selected_color = color_var.get()
    if selected_color not in color_ranges:
        print("Invalid color selected!")
        return

    running = True
    threading.Thread(target=detection_loop).start()

# Stop button callback
def stop_detection():
    global running
    running = False

# GUI setup
def setup_gui():
    global color_var

    root = tk.Tk()
    root.title("Enemy Detector")

    # Dropdown menu for color selection
    tk.Label(root, text="Select Enemy Color:").pack()
    color_var = StringVar()
    color_var.set("Red")  # Default value
    tk.OptionMenu(root, color_var, *color_ranges.keys()).pack()

    # Start button
    tk.Button(root, text="Start Detection", command=start_detection).pack()

    # Stop button
    tk.Button(root, text="Stop Detection", command=stop_detection).pack()

    # Quit button
    tk.Button(root, text="Quit", command=root.quit).pack()

