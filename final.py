# Holographic Internal Body Examiner
# Zorawar Kamboj
# July 8, 2026
# This project detects people's bodies in realtime and holographics form to examine people's internal skelton bones and organs

# import the packages
import cv2
import torch
import numpy as np
import time
import sys
from ultralytics import YOLO

# add the device type if gpu is available use it, otherwise use the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# add the YOLO26 segmentation file
print("Loading YOLO26 segmentation model...")
model = YOLO("yolo26n-seg.pt")
print("Model loaded!")

# define the ultrasounding function
def ultrasound_filter(gray, noise_level=0.12, contrast=2.2):
    tensor = torch.from_numpy(gray).float().to(device) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    noise = torch.randn_like(tensor) * noise_level + 1.0
    noisy = torch.clamp(tensor * noise, 0, 1)
    noisy = noisy.squeeze().cpu().numpy() * 255
    noisy = noisy.astype(np.uint8)
    blurred = cv2.GaussianBlur(noisy, (5, 5), 1.2)

    tensor = torch.from_numpy(blurred).float().to(device) / 255.0
    p_low = torch.quantile(tensor, 0.02)
    p_high = torch.quantile(tensor, 0.98)
    enhanced = torch.clamp((tensor - p_low) / (p_high - p_low + 1e-6), 0, 1)
    enhanced = enhanced.pow(1.0 / contrast)

    result = (enhanced * 255).byte().cpu().numpy()
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

# define the holographic body scanner function
def holographic_scan(us_frame, mask, frame_count):
    h, w = us_frame.shape[:2]
    holo = np.zeros_like(us_frame)

    cv2.ellipse(holo, (w//2, int(h*0.45)), (65, 48), 0, 0, 360, (0, 255, 200), -1)
    cv2.line(holo, (w//2, int(h*0.45)+60), (w//2, h-40), (0, 255, 180), 10)

    for y in range(0, h, 35):
        cv2.line(holo, (0, y), (w, y), (0, 180, 120), 1)
    for x in range(0, w, 35):
        cv2.line(holo, (x, 0), (x, h), (0, 180, 120), 1)

    beam_x = int(w * (0.3 + 0.4 * np.sin(frame_count * 0.08)))
    cv2.line(holo, (beam_x, 0), (beam_x, h), (0, 255, 255), 3)

    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    holo = cv2.bitwise_and(holo, mask3)

    glow = cv2.GaussianBlur(holo, (15, 15), 0)
    holo = cv2.addWeighted(holo, 1.0, glow, 0.8, 0)

    final = cv2.addWeighted(us_frame, 0.6, holo, 0.8, 0)
    return final

# define the organs examiner function
def create_organs_view(mask, frame_count):
    h, w = mask.shape
    internal = np.zeros((h, w, 3), dtype=np.uint8)

    # Brain
    cv2.ellipse(internal, (w//2, int(h*0.22)), (70, 50), 0, 0, 360, (100, 100, 255), -1)
    cv2.putText(internal, "BRAIN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Heart
    cv2.ellipse(internal, (w//2 - 20, int(h*0.42)), (48, 40), -35, 0, 360, (0, 100, 255), -1)
    cv2.putText(internal, "HEART", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Lungs
    cv2.ellipse(internal, (w//2 - 80, int(h*0.38)), (40, 50), 0, 0, 360, (0, 255, 180), -1)
    cv2.ellipse(internal, (w//2 + 80, int(h*0.38)), (40, 50), 0, 0, 360, (0, 255, 180), -1)
    cv2.putText(internal, "LUNGS", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Liver
    cv2.ellipse(internal, (w//2 + 60, int(h*0.65)), (70, 42), 30, 0, 360, (0, 180, 100), -1)
    cv2.putText(internal, "LIVER", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Stomach
    cv2.ellipse(internal, (w//2 - 55, int(h*0.62)), (48, 35), -20, 0, 360, (0, 200, 120), -1)
    cv2.putText(internal, "STOMACH", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Kidneys
    cv2.ellipse(internal, (w//2 - 100, int(h*0.68)), (28, 20), 0, 0, 360, (0, 220, 150), -1)
    cv2.ellipse(internal, (w//2 + 115, int(h*0.68)), (28, 20), 0, 0, 360, (0, 220, 150), -1)
    cv2.putText(internal, "KIDNEYS", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    glow = cv2.GaussianBlur(internal, (25, 25), 0)
    internal = cv2.addWeighted(internal, 1.0, glow, 0.9, 0)

    return internal

# define the skeleton examiner function
def create_skeleton_view(mask, frame_count):
    h, w = mask.shape
    skeleton = np.zeros((h, w, 3), dtype=np.uint8)

    # Cranium
    cv2.ellipse(skeleton, (w//2, int(h*0.18)), (75, 55), 0, 0, 360, (220, 220, 255), -1)
    cv2.putText(skeleton, "CRANIUM", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Eyes
    cv2.circle(skeleton, (w//2 - 25, int(h*0.18)), 12, (0, 200, 255), -1)
    cv2.circle(skeleton, (w//2 + 25, int(h*0.18)), 12, (0, 200, 255), -1)

    # Spine & Rib Cage
    cv2.line(skeleton, (w//2, int(h*0.32)), (w//2, h-70), (220, 220, 255), 16)
    for i in range(-4, 5):
        cv2.line(skeleton, (w//2 - 75, int(h*0.4) + i*18), (w//2 + 75, int(h*0.4) + i*18), (200, 200, 255), 6)
    cv2.putText(skeleton, "SPINE & RIBS", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Arms
    cv2.line(skeleton, (w//2 - 90, int(h*0.45)), (w//2 - 150, int(h*0.75)), (200, 200, 255), 10)
    cv2.line(skeleton, (w//2 + 90, int(h*0.45)), (w//2 + 150, int(h*0.75)), (200, 200, 255), 10)
    cv2.putText(skeleton, "ARMS", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Hands
    cv2.rectangle(skeleton, (w//2 - 155, int(h*0.75)), (w//2 - 170, int(h*0.82)), (200, 200, 255), -1)
    cv2.rectangle(skeleton, (w//2 + 155, int(h*0.75)), (w//2 + 170, int(h*0.82)), (200, 200, 255), -1)
    cv2.putText(skeleton, "HANDS", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Legs
    cv2.line(skeleton, (w//2 - 40, h-90), (w//2 - 55, h-20), (200, 200, 255), 12)
    cv2.line(skeleton, (w//2 + 40, h-90), (w//2 + 55, h-20), (200, 200, 255), 12)
    cv2.putText(skeleton, "LEGS", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Feet
    cv2.rectangle(skeleton, (w//2 - 65, h-25), (w//2 - 45, h-15), (200, 200, 255), -1)
    cv2.rectangle(skeleton, (w//2 + 45, h-25), (w//2 + 65, h-15), (200, 200, 255), -1)
    cv2.putText(skeleton, "FEET", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    glow = cv2.GaussianBlur(skeleton, (25, 25), 0)
    skeleton = cv2.addWeighted(skeleton, 1.0, glow, 0.9, 0)

    return skeleton

# define the main function
def main():
    # Load the model and set up the video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Show all the windows
    cv2.namedWindow('Holographic Body Scan', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Holographic Body Scan', 1280, 720)

    cv2.namedWindow('Organs Examination', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Organs Examination', 640, 480)

    cv2.namedWindow('Skeleton Examination', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Skeleton Examination', 640, 480)

    cv2.createTrackbar('Noise', 'Holographic Body Scan', 12, 40, lambda x: None)
    cv2.createTrackbar('Contrast', 'Holographic Body Scan', 22, 40, lambda x: None)

    prev_time = time.time()
    frame_count = 0

    print("🎥 Separate Organs & Skeleton windows with clear labels!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        noise_val = cv2.getTrackbarPos('Noise', 'Holographic Body Scan') / 100.0
        contrast_val = cv2.getTrackbarPos('Contrast', 'Holographic Body Scan') / 10.0

        results = model(frame, classes=0, verbose=False)

        us_frame = frame.copy()
        person_detected = False
        body_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    mask = mask.cpu().numpy().astype(np.uint8)
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    body_mask = cv2.bitwise_or(body_mask, mask)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    body_us = ultrasound_filter(gray, noise_level=noise_val, contrast=contrast_val)
                    us_frame = np.where(mask[..., None] > 0, body_us, us_frame)
                    person_detected = True

        # if a person if detected, show the holographic scan
        if person_detected:
            us_frame = holographic_scan(us_frame, body_mask, frame_count)
            contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(us_frame, contours, -1, (0, 255, 255), 5)

        h, w = us_frame.shape[:2]
        line_pos = (frame_count * 15) % w
        cv2.line(us_frame, (line_pos, 0), (line_pos, h), (0, 255, 255), 2)

        # calculate and display the FPS
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(us_frame, f"FPS: {fps:.1f} | RTX 5090 HOLO", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 200), 3)

        cv2.imshow('Holographic Body Scan', us_frame)

        # if a person is detected, show the organs and skeleton views
        if person_detected:
            organs_view = create_organs_view(body_mask, frame_count)
            skeleton_view = create_skeleton_view(body_mask, frame_count)
            cv2.imshow('Organs Examination', organs_view)
            cv2.imshow('Skeleton Examination', skeleton_view)
        # otherwise, show the black screens for the organs and skeleton views
        else:
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black, "No Person Detected", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3)
            cv2.imshow('Organs Examination', black)
            cv2.imshow('Skeleton Examination', black)

        frame_count += 1

        # if the 'q' is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

# run the main function
if __name__ == "__main__":
    main()