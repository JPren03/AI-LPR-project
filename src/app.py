import cv2
import numpy as np
import re
import os
import easyocr
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

def crop_from_bbox(img, bbox, pad=0):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

def clear_directories():
    out_dir = Path(BASE_DIR / "outputs/plates")
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    out_dir = Path(BASE_DIR / "outputs/heatmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

def show_detection_heatmap(image, results):
    if results is None:
        return
    
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return

    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        heat[y1:y2, x1:x2] += c

    heat = cv2.GaussianBlur(heat, (51, 51), 0)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    heatmap = np.uint8(255 * heat)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    out_dir = Path(BASE_DIR / "outputs/heatmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = count = len(list(out_dir.glob("*.jpg")))
    heatname = f"Heatmap{count+1}.jpg"
    out_path = out_dir / heatname

    cv2.imwrite(out_path, overlay)

# Vehicle Detection
def vehicle_detection(vehicle_model, image, class_list, VEHICLE_CLASSES):
    vehicle_results = vehicle_model.predict(image, conf = 0.4, device="cpu", verbose = False)
    show_detection_heatmap(image, vehicle_results)
    detections = vehicle_results[0].boxes.data.detach().cpu().numpy()

    vehicle_crops = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        confidence = float(det[4])
        class_id = int(det[5])
        class_name = class_list[class_id]

        if class_name in VEHICLE_CLASSES:
            v_crop = crop_from_bbox(image, (x1, y1, x2, y2), pad=5)
            if v_crop is None:
                continue

            vehicle_crops.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": {x1, y1, x2, y2},
                "crop": v_crop
            })
    return vehicle_crops

# Plate Detection
def plate_detection(plate_model, vehicle_crops):
    plate_crops = []
    for vi, v in enumerate(vehicle_crops):
        v_img = v["crop"]

        plate_results = plate_model.predict(
            v_img,
            conf=0.25, 
            iou = 0.45, 
            device ="cpu", 
            verbose = False)
        show_detection_heatmap(v_img, plate_results)

        r = plate_results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        
        order = np.argsort(-confs) 
        for pj, idx in enumerate(order):
            px1, py1, px2, py2 = xyxy[idx].astype(int)
            pconf = float(confs[idx])

            p_crop = crop_from_bbox(v_img, (px1, py1, px2, py2), pad=5)
            if p_crop is None:
                continue

            plate_crops.append({
                "vehicle_index": vi,
                "plate_index_in_vehicle": pj,
                "plate_bbox_in_vehicle": (px1, py1, px2, py2),
                "plate_conf": pconf,
                "crop": p_crop
            })

        out_dir = Path(BASE_DIR / "outputs/plates")
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

        for i, p in enumerate(plate_crops):
            out_path = out_dir / f"plate_{i + 1}.jpg"
            cv2.imwrite(str(out_path), p["crop"])
    

#Character Detection and Recognition

def run_reader(reader, image):
    results = reader.readtext(image, detail=1, paragraph = False)

    text_parts = [r[1] for r in results]
    joined_text = " ".join(text_parts).strip()

    confidences = [r[2] for r in results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": joined_text,
        "confidence": avg_conf,
        "raw": results
    }

def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray

def clean_english_text(text):

    if not text:
        return ""

    text = text.upper()

    # remove spaces
    text = text.replace(" ", "")

    # keep only English letters and digits
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text

def clean_arabic_text(text):

    if not text:
        return ""

    text = text.replace(" ", "")

    # keep ONLY Arabic letters and Arabic digits
    text = re.sub(r"[^ء-ي٠-٩]", "", text)

    return text

def validate_egypt_plate(text):
    if not text:
        return "", "Unknown Plate Type"

    cleaned = re.sub(r"\s+", "", text)
    cleaned = cleaned.replace("\u200f", "").replace("\u200e", "")

    if "مصر" in cleaned or "رصم" in cleaned:
        cleaned = cleaned.replace("مصر", "").replace("رصم", "")
        return cleaned, "Egyptian Plate Type"

    cleaned = re.sub(r"[^ء-يأإآ٠-٩]", "", cleaned)

    letters = len(re.findall(r"[ء-يأإآ]", cleaned))
    digits = len(re.findall(r"[٠-٩]", cleaned))

    candidates = [cleaned, cleaned[::-1]]
    for c in candidates:
        if re.fullmatch(r"^[ء-يأإآ]{3}[٠-٩]{3,4}$", c) or re.fullmatch(r"^[٠-٩]{3,4}[ء-يأإآ]{3}$", c):
            return c, "Egyptian Plate Type"

    # stricter relaxed fallback
    if letters == 3 and digits in [3, 4]:
        return cleaned, "Probable Egyptian Plate"

    return cleaned, "Unknown Plate Type"

def validate_uk_plate(text):
    LETTER_TO_DIGIT = {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }

    DIGIT_TO_LETTER = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B"
    }
    
    verdict = "Unknown Plate Type"
    if not text:
        return "", verdict

    cleaned = re.sub(r"\s+", "", text).upper()

    if "GB" in cleaned:
        verdict = "Probable UK Plate"
    if len(cleaned) < 7 or len(cleaned) > 9:
        return cleaned, verdict

    strict_pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"

    # Strict match
    if re.fullmatch(strict_pattern, cleaned):
        verdict = "UK Plate Type"
        return cleaned, verdict

    # Try OCR-style correction
    expected_pattern = ["L", "L", "D", "D", "L", "L", "L"]
    corrected = []

    for ch, expected in zip(cleaned, expected_pattern):
        if expected == "L":
            if ch.isalpha():
                corrected.append(ch)
            elif ch.isdigit() and ch in DIGIT_TO_LETTER:
                corrected.append(DIGIT_TO_LETTER[ch])
            else:
                return cleaned, verdict

        elif expected == "D":
            if ch.isdigit():
                corrected.append(ch)
            elif ch.isalpha() and ch in LETTER_TO_DIGIT:
                corrected.append(LETTER_TO_DIGIT[ch])
            else:
                return cleaned, verdict

    corrected_text = "".join(corrected)

    if re.fullmatch(strict_pattern, corrected_text):
        verdict = "UK Plate Type"
        return corrected_text, verdict

    return cleaned, verdict

def score_candidate(candidate):
    lang = candidate["language"]
    text = candidate["result"]["text"]
    conf = candidate["result"]["confidence"]

    if lang == "arabic":
        corrected, plate_type = validate_egypt_plate(text)
    elif lang == "english":
        corrected, plate_type = validate_uk_plate(text)
    else:
        corrected, plate_type = text, "Unknown Plate Type"

    candidate["validated_text"] = corrected
    candidate["plate_type"] = plate_type

    if "Plate Type" in plate_type and "Unknown" not in plate_type:
        bonus = 3.0
    elif "Probable" in plate_type:
        bonus = 1.5
    else:
        bonus = 0.0

    length_bonus = min(len(corrected), 7) * 0.05

    return bonus + conf + length_bonus

def ocr_heatmap(image, ocr_raw):
    if image is None:
        return

    if not ocr_raw:
        print("No OCR boxes for heatmap")
        return

    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    
    for item in ocr_raw:
        bbox = item[0]
        conf = float(item[2]) if len(item) > 2 else 1.0

        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(heat, [pts], conf)

    heat = cv2.GaussianBlur(heat, (31, 31), 0)

    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    heatmap = np.uint8(255 * heat)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    out_dir = Path(BASE_DIR / "outputs/heatmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = count = len(list(out_dir.glob("*.jpg")))
    heatname = f"Heatmap{count+1}.jpg"
    out_path = out_dir / heatname

    cv2.imwrite(out_path, overlay)


def OCR_main():
    plate_dir = Path(BASE_DIR / "outputs/plates")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not plate_dir.exists():
        print("Folder not found:", plate_dir.resolve())
        return

    
    english_reader = easyocr.Reader(["en"])
    arabic_reader = easyocr.Reader(["ar"])

    for img_path in sorted(plate_dir.iterdir()):
        if img_path.suffix.lower() not in valid_exts:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print("Failed to load:", img_path.name)
            continue

        processed = preprocess_plate(image)

        #English
        english_original = run_reader(english_reader, image)
        english_original["text"] = clean_english_text(english_original["text"])

        english_processed = run_reader(english_reader, processed)
        english_processed["text"] = clean_english_text(english_processed["text"])

        #Arabic
        arabic_original = run_reader(arabic_reader, image)
        arabic_original["text"] = clean_arabic_text(arabic_original["text"])

        arabic_processed = run_reader(arabic_reader, processed)
        arabic_processed["text"] = clean_arabic_text(arabic_processed["text"])

        candidates = [
            {
                "language": "english",
                "image_version": "original",
                "result": english_original
            },
            {
                "language": "english",
                "image_version": "processed",
                "result": english_processed
            },
            {
                "language": "arabic",
                "image_version": "original",
                "result": arabic_original
            },
            {
                "language": "arabic",
                "image_version": "processed",
                "result": arabic_processed
            }
        ]

        best = max(candidates, key=score_candidate)

        best_image = image if best["image_version"] == "original" else processed
        ocr_heatmap(best_image, best["result"]["raw"])
        
        rawOCR = best["result"]["text"]
        platetype = best.get("plate_type", "Unknown Plate Type")
        best["result"]["text"] = best.get("validated_text", rawOCR)

        print(f"File: {img_path.name}")
        print(f"Language: {best['language']}")
        print(f"Image version: {best['image_version']}")
        print(f"Contents: {best['result']['text'] if best['result']['text'] else '[nothing found]'}")
        print(f"Confidence: {best['result']['confidence']:.3f}")
        print(f"Plate Type: {platetype}")
        print(f"Raw OCR: {rawOCR}")
        print()


##Main
def main():

    clear_directories()
    
    vehicle_model   = YOLO("yolo26n.pt")
    plate_model     = YOLO(BASE_DIR / "plateDetectModel/best.pt")
    class_list = vehicle_model.names
    VEHICLE_CLASSES = {
        'car',
        'motorcycle',
        'bus',
        'truck'   # includes vans & lorries in COCO
    }

    image_path = BASE_DIR / "data/LPRdatasets/plateImages/original.jpg"
    image = cv2.imread(image_path)
    if image is None: 
        raise FileNotFoundError(image_path)
    
    vehicle_crops = vehicle_detection(vehicle_model, image, class_list, VEHICLE_CLASSES)
    plate_detection(plate_model, vehicle_crops)
    OCR_main()

if __name__ == "__main__":
    main()


