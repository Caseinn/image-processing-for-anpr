# anpr.py
import cv2
import easyocr
import os
import csv
from datetime import datetime
from utils import (
    preprocess_for_detection,
    find_plate_candidates,
    clean_ocr_text,
    is_valid_indonesian_plate,
    debug_pipeline
)
from config import (
    INPUT_VIDEO,
    OUTPUT_VIDEO,
    CSV_OUTPUT,
    CROPPED_PLATES_DIR,
    DEBUG_FRAME,
    INDO_PLATE_PATTERN
)

class ANPRSystem:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.recent_plates = {}
        self.frame_count = 0

    def process_video(self, show_video=False):
        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            raise FileNotFoundError(f"❌ Cannot open video: {INPUT_VIDEO}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
        os.makedirs(CROPPED_PLATES_DIR, exist_ok=True)

        csv_file = open(CSV_OUTPUT, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'timestamp_sec', 'plate_text', 'cropped_image'])

        print("🚀 Starting ANPR processing...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            timestamp = self.frame_count / fps

            edged, gray = preprocess_for_detection(frame)
            frame_area = width * height
            candidates = find_plate_candidates(edged, frame_area)

            detected_plate = None
            cropped_path = ""
            current_plate_roi = None
            valid_x, valid_y, valid_w, valid_h = 0, 0, 0, 0

            for (x, y, w, h) in sorted(candidates, key=lambda r: r[2]*r[3], reverse=True):
                plate_roi = frame[y:y+h, x:x+w]
                if plate_roi.size == 0 or w < 30 or h < 15:
                    continue

                try:
                    ocr_results = self.reader.readtext(plate_roi, detail=0, paragraph=False, min_size=10)
                except Exception as e:
                    print(f"⚠️ OCR error: {e}")
                    continue

                if not ocr_results:
                    continue

                raw_text = ''.join(ocr_results)
                clean_text = clean_ocr_text(raw_text)

                if is_valid_indonesian_plate(clean_text, INDO_PLATE_PATTERN):
                    current_frame = self.frame_count
                    if clean_text in self.recent_plates and (current_frame - self.recent_plates[clean_text]) < 15:
                        continue

                    self.recent_plates[clean_text] = current_frame
                    detected_plate = clean_text
                    current_plate_roi = plate_roi.copy()
                    valid_x, valid_y, valid_w, valid_h = x, y, w, h

                    plate_filename = f"plate_{self.frame_count}_{clean_text}.jpg"
                    cropped_path = os.path.join(CROPPED_PLATES_DIR, plate_filename)
                    cv2.imwrite(cropped_path, plate_roi)

                    csv_writer.writerow([self.frame_count, f"{timestamp:.2f}", clean_text, plate_filename])
                    print(f"✅ Frame {self.frame_count}: {clean_text}")
                    break

            # Debug
            if DEBUG_FRAME > 0 and self.frame_count == DEBUG_FRAME:
                debug_pipeline(frame, edged, candidates, current_plate_roi, self.frame_count)

            # Annotate
            if detected_plate:
                cv2.rectangle(frame, (valid_x, valid_y), (valid_x + valid_w, valid_y + valid_h), (0, 255, 0), 2)
                cv2.putText(frame, detected_plate, (valid_x, valid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write to output video
            out.write(frame)

            if show_video:
                cv2.imshow('ANPR Live', frame)
                # Press 'q' to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Stopped by user.")
                    break

        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()

        print("\n🎉 Processing complete!")
        print(f"📁 Output video: {OUTPUT_VIDEO}")
        print(f"📄 CSV results: {CSV_OUTPUT}")
        print(f"🖼️  Cropped plates: {CROPPED_PLATES_DIR}")

        if DEBUG_FRAME > 0:
            print(f"🔍 Debug images saved in: debug/frame_{DEBUG_FRAME}/")