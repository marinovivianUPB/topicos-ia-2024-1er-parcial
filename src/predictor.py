from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings
from PIL import Image

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    segment_polygon = Polygon(segment)
    min_distance = max_distance

    for bbox in bboxes:
        weapon_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
        distance = segment_polygon.distance(weapon_box)

        if distance < min_distance:
            min_distance = distance
            matched_box = bbox

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()
    overlay=image_array.copy()

    for label, polygon, box in zip(segmentation.labels, segmentation.polygons, segmentation.boxes):
        color = (0, 255, 0) if label == "safe" else (255, 0, 0)

        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [pts], color=color)

        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.addWeighted(overlay, 0.4, annotated_img, 1, 0, annotated_img)

    return annotated_img

class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)
        print(f"loading seg model: {SETTINGS.custom_model_path}")
        self.custom_model = YOLO(SETTINGS.custom_model_path)

    def detect_guns(self, image_array, threshold: float = 0.5, model:int = 1):
        if model == 1:
            results = self.od_model(image_array, conf=threshold)[0]
            labels = results.boxes.cls.tolist()
            indexes = [
                i for i in range(len(labels)) if labels[i] in [3, 4] #[0]
            ]  # 0 = "person"
        else:
            results = self.custom_model(image_array, conf=threshold)[0]
            labels = results.boxes.cls.tolist()
            indexes = [
                i for i in range(len(labels)) if labels[i] in [0,1]
            ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        seg_results = self.seg_model(image_array, conf=threshold)[0]
    
        labels = seg_results.boxes.cls.tolist()
        indexes = [i for i in range(len(labels)) if labels[i] == 0] #porque se definio 0 en detect_guns 

        polygons = [seg_results.masks.xy[i] for i in indexes]
        #Segmentation exige que los valores de polygons sean enteros
        polygons = [[[int(coordenates[0]), int(coordenates[1])] for coordenates in polygon] for polygon in polygons]

        boxes = [seg_results.boxes.xyxy[i].tolist() for i in indexes]
        #Segmentation exige que los valores de boxes sean enteros
        boxes = [[int(x) for x in box] for box in boxes]

        other_labels = []

        gun_boxes = self.detect_guns(image_array, threshold).boxes

        for poly in polygons:
            matched_box = match_gun_bbox(poly, gun_boxes, max_distance)
            if matched_box:
                other_labels.append("danger")
            else:
                other_labels.append("safe")

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(polygons),
            polygons=polygons,
            boxes=boxes,
            labels=other_labels
        )
