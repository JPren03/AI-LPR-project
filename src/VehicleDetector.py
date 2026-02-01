from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

        # COCO class names
        self.class_list = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

        # Vehicle classes
        self.vehicle_classes = {
            'car',
            'motorcycle',
            'bus',
            'truck'   # vans included here
        }

    def detect(self, image):
        """
        Detect vehicles in a single image.
        """

        results = self.model.predict(
            image,
            conf=self.conf,
            verbose=False
        )

        detections = results[0].boxes.data.detach().cpu().numpy()
        h, w = image.shape[:2]

        vehicles = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            confidence = float(det[4])
            class_id = int(det[5])
            class_name = self.class_list[class_id]

            if class_name not in self.vehicle_classes:
                continue

            # Clamp coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            vehicles.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "crop": crop
            })

        return vehicles
