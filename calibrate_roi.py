import cv2
import json
import sys
from pathlib import Path

# ใช้: python calibrate_roi.py test2.jpg
if len(sys.argv) < 2:
    print("Usage: python calibrate_roi.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
img = cv2.imread(image_path)
if img is None:
    print(f"Cannot read image: {image_path}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"Loaded image size: width={w}, height={h}")

# ชื่อ ROI ที่เราจะกำหนด
roi_order = [
    ("roi_b", "UPC / MODEL line (บนสุด)"),
    ("roi_a", "Serial Number"),
    ("roi_c", "Part Number / BATCH"),
]

current_idx = 0
points = []  # เก็บ 2 จุด (x1,y1,x2,y2)
roi_config = {}

window_name = "Calibrate ROI (left click 2 points per ROI, press 'n' for next, 'q' to quit)"
cv2.namedWindow(window_name)

def mouse_callback(event, x, y, flags, param):
    global points, current_idx, roi_config, img

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked at x={x}, y={y}")

        # วาดจุดบนภาพเพื่อให้เห็น
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # ถ้าได้ 2 จุดแล้ว -> คำนวณ ROI
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            w_roi = x_max - x_min
            h_roi = y_max - y_min

            roi_name, desc = roi_order[current_idx]
            roi_config[roi_name] = {
                "label": "S/N" if roi_name == "roi_a"
                         else "MODEL" if roi_name == "roi_b"
                         else "BATCH",
                "x": int(x_min),
                "y": int(y_min),
                "w": int(w_roi),
                "h": int(h_roi),
            }

            print(f"\n[{roi_name}] {desc}")
            print(f"  x={x_min}, y={y_min}, w={w_roi}, h={h_roi}\n")

            # วาดสี่เหลี่ยมให้เห็น
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            points = []  # reset สำหรับ ROI ถัดไป

cv2.setMouseCallback(window_name, mouse_callback)

print("\nInstructions:")
print(" - ตามลำดับ ROI ที่จะแสดงบน console")
print(" - คลิกซ้าย 2 ครั้งต่อ 1 ROI (มุมซ้ายบน & มุมขวาล่าง)")
print(" - เมื่อพอใจกับ ROI ปัจจุบันแล้ว กดปุ่ม 'n' เพื่อไป ROI ถัดไป")
print(" - กด 'q' เพื่อบันทึกและออก\n")

while True:
    # copy image เพื่อไม่ให้ทับวาดซ้ำ (หรือจะใช้ img เดิมก็ได้ ถ้าคุณโอเค)
    display = img.copy()
    roi_name, desc = roi_order[current_idx]
    cv2.putText(
        display,
        f"Current: {roi_name} ({desc})  |  press 'n' for next, 'q' to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.imshow(window_name, display)
    key = cv2.waitKey(50) & 0xFF

    if key == ord("n"):
        # ไป ROI ถัดไป (ถ้ามี)
        if current_idx < len(roi_order) - 1:
            current_idx += 1
            print(f"\n>>> Switch to {roi_order[current_idx][0]} : {roi_order[current_idx][1]}")
        else:
            print("\n>>> Already at last ROI. Press 'q' to save & quit.")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

# เซฟลง config/roi_config.json
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)
config_path = config_dir / "roi_config.json"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(roi_config, f, indent=2)

print("\nSaved ROI config to:", config_path)
print(json.dumps(roi_config, indent=2))