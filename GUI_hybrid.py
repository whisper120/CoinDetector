import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from rfdetr import RFDETRBase

# Global variables for models
yolo_region_model = YOLO("models/Latest_ROI_yolo11.pt")
letter_model = RFDETRBase(pretrain_weights="models/Latest_Letter_RFDETR.pth")

HEBREW_LETTER_VALUES = {
    "alef": 1,
    "bet": 2,
    "gimel": 3,
    "daled": 4,
    "hey": 5,
    "vav": 6,
    "zayin": 7,
    "heth": 8,
    "tet": 9,
    "mem": 40,
    "noon": 50,
    "sameh": 60,
    "hain": 70,
    "pay": 80,
    "shin": 300,
    "thaf": 400
}

class_names = {
    1: "alef",
    2: "bet",
    3: "daled",
    4: "gimel",
    5: "hain",
    6: "heth",
    7: "hey",
    8: "mem",
    9: "noon",
    10: "pay",
    11: "sameh",
    12: "shin",
    13: "tet",
    14: "thaf",
    15: "vav",
    16: "zayin"
}


def hebrew_letters_to_civil_year(letters):
    total = 0
    hey_used_as_5000 = False

    for letter in letters:
        lower_letter = letter.lower()
        if lower_letter == "hey" and not hey_used_as_5000:
            total += 5000
            hey_used_as_5000 = True
        else:
            value = HEBREW_LETTER_VALUES.get(lower_letter)
            if value is None:
                raise ValueError(f"Unknown letter: {letter}")
            total += value

    civil_year = total - 3760
    return civil_year


def process_image(image_path=None, frame=None):
    """
    Processes an image (either from file path or a captured frame)
    and updates the information output.
    """
    if image_path:
        # Load image from file path
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", f"Could not load image from {image_path}")
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for processing consistent with webcam
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
            return

    if frame is None:
        return # No image to process

    detected_letters = []

    # Run region detection model
    region_results = yolo_region_model(frame)
    if region_results:
        region_result = region_results[0]
        save_folder = "processed_images" # Create a folder for processed images
        os.makedirs(save_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, box in enumerate(region_result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clip coordinates to image size
            h, w, _ = frame.shape
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Save cropped ROI (optional, for debugging/record)
            crop_filename = f"crop_{i}_{timestamp}.png"
            crop_path = os.path.join(save_folder, crop_filename)
            cv2.imwrite(crop_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)) # Save as BGR

            # Convert ROI to PIL for RFDETR
            roi_pil = Image.fromarray(roi).convert("RGB")

            # Run letter detection using RFDETR
            detections = letter_model.predict(roi_pil)

            # Save annotation (optional)
            annotation_path = os.path.join(save_folder, f"{crop_filename}.txt")
            with open(annotation_path, "w", encoding="utf-8") as f:
                for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                    score = float(score)
                    class_id = int(class_id)
                    class_name = class_names.get(class_id, f"Class {class_id}")

                    if score >= accuracy_value.get():
                        detected_letters.append(class_name)
                        f.write(f"{class_name} {score:.2f}\n")

    # Display results in the GUI
    output_text = (
        "Detected Hebrew Letters: " + ", ".join(detected_letters) + " --> Year " + str(hebrew_letters_to_civil_year(detected_letters))
        if detected_letters else "No Hebrew letters detected."
    )
    if detected_letters:
        if hebrew_letters_to_civil_year(detected_letters) > 2026:
            output_text += "\nAccuracy might be too low and produced irrelevant results."
    info_output.delete(1.0, tk.END)
    info_output.insert(tk.END, output_text)


    years_to_notify = [int(y.strip()) for y in years_entry.get().split(',') if y.strip().isdigit()]
    if detected_letters:
        try:
            # Convert detected letters into a civil year
            detected_year = hebrew_letters_to_civil_year(detected_letters)

            # Check if the detected year is in the list
            if detected_year in years_to_notify:
                info_output.insert(tk.END, f"\nRare Coin Notification: Detected 10 Agorot dated {detected_year}!")
                messagebox.showinfo("Rare Coin Detected", f"Detected 10 Agorot dated {detected_year}!")
            else:
                info_output.insert(tk.END, f"\nDetected year: {detected_year} (not in notification list).")

        except ValueError as e:
            info_output.insert(tk.END, f"\nError decoding letters: {e}")
    else:
        info_output.insert(tk.END, "\nNo specific year detected or matched for notification.")


def take_picture():
    global cap, live_label # Ensure cap and live_label are accessible
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open webcam.")
        return

    capture_window = tk.Toplevel(root)
    capture_window.title("Camera Preview")
    capture_window.geometry("640x550") # Adjust size to fit video and button

    live_label = tk.Label(capture_window)
    live_label.pack(pady=10)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            live_label.imgtk = imgtk
            live_label.configure(image=imgtk)
            live_label.after(10, update_frame)
        else:
            messagebox.showerror("Camera Error", "Failed to capture frame from webcam.")
            cap.release() # Release camera if capture fails
            capture_window.destroy()

    def check_webcam_picture():
        ret, frame = cap.read()
        if ret:
            # Pass the captured frame directly for processing
            process_image(frame=frame)
        else:
            messagebox.showerror("Capture Failed", "Failed to capture image from webcam.")
        cap.release() # Release camera after taking picture
        capture_window.destroy() # Close webcam window

    capture_button = tk.Button(capture_window, text="Check", command=check_webcam_picture)
    capture_button.pack(pady=5)

    update_frame() # Start showing frames

    # Handle window closing to release camera
    capture_window.protocol("WM_DELETE_WINDOW", lambda: [cap.release(), capture_window.destroy()])


def choose_picture_from_pc():
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*"))
    )
    if file_path:
        process_image(image_path=file_path)

# Main window
root = tk.Tk()
root.title("Coin Detection")
root.geometry("650x600") # Increased size to accommodate new elements

# Take Picture
take_picture_label = tk.Label(root, text="Take Picture from Webcam")
take_picture_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')

take_picture_button = tk.Button(root, text="Take Picture", command=take_picture)
take_picture_button.grid(row=0, column=1, padx=10, pady=5)

# Choose Picture from PC
choose_picture_label = tk.Label(root, text="Choose Picture from PC")
choose_picture_label.grid(row=1, column=0, padx=10, pady=5, sticky='w') # Adjusted row to 1

choose_picture_button = tk.Button(root, text="Choose Picture", command=choose_picture_from_pc)
choose_picture_button.grid(row=1, column=1, padx=10, pady=5) # Adjusted row to 1

# Set Accuracy (Slider)
accuracy_label = tk.Label(root, text="Set Accuracy (0 to 1)")
accuracy_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')

accuracy_value = tk.DoubleVar(value=0.5)
accuracy_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=accuracy_value)
accuracy_slider.grid(row=2, column=1, padx=10, pady=5)

# Enter Years
years_label = tk.Label(root, text="Enter Years (comma separated)")
years_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')

years_entry = tk.Entry(root, width=30)
years_entry.insert(0, "2007,2009")
years_entry.grid(row=3, column=1, padx=10, pady=5)

# Information Output Title
info_title = tk.Label(root, text="Information Output", font=('Helvetica', 12, 'bold'))
info_title.grid(row=4, column=0, columnspan=2, pady=10)

# Information Output Box
info_output = tk.Text(root, height=10, width=70) # Increased height and width for more info
info_output.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Placeholder for detected coins/letters from process_image
info_output.insert(tk.END, "No detection performed yet. Take or choose a picture.")


root.mainloop()