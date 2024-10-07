import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from shapely.geometry import Polygon
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.config import get_settings
from src.models import Gun, Person, GunType, PersonType

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold, model:int = 1) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    #poner img_obj para trabajar con modelo propio
    if model == 1:
        return detector.detect_guns(img_array, threshold), img_array
    else:
        return detector.detect_guns(img_obj, threshold, model), img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    model: int = 1,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold,model)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    model: int = 1,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold, model)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    img_stream = io.BytesIO(file.file.read())
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    segmentation = detector.segment_people(img_array, threshold)
    
    return segmentation


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    draw_boxes: bool = False,
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    img_stream = io.BytesIO(file.file.read())
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    segmentation = detector.segment_people(img_array, threshold)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect")
def detect(
    threshold: float = 0.5,
    model: int = 1,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> dict:
    detection, img = detect_uploadfile(detector, file, threshold,model)
    segmentation = detector.segment_people(img, threshold)

    return {
        "detection": detection,
        "segmentation": segmentation
    }


@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    model: int = 1,
    file: UploadFile = File(...),
    draw_boxes: bool = False,
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold, model)
    segmentation = detector.segment_people(img, threshold)

    annotated_img = annotate_detection(img, detection)
    annotated_img = annotate_segmentation(annotated_img, segmentation, draw_boxes)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/guns")
def guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    model: int = 1,
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:
    detection, _ = detect_uploadfile(detector, file, threshold,model)
    guns = []
    for label, box in zip(detection.labels, detection.boxes):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        gun = Gun(
            gun_type=GunType(label.lower()),
            location={"x": center_x, "y": center_y}
        )
        guns.append(gun)
    return guns


@app.post("/people")
def people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:
    img_stream = io.BytesIO(file.file.read())
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    segmentation = detector.segment_people(img_array, threshold)
    people = []
    for label, poly in zip(segmentation.labels, segmentation.polygons):
        #los polígonos irregulares no tienen un centro
        #estamos considerando como centro los promedios de x y y
        #se considera que la mejor opción hubiera sido devolver los
        #centros de los boxes, pero la instruccion
        #pedia el centro de los segmentos
        center_x = int(np.mean([point[0] for point in poly]))
        center_y = int(np.mean([point[1] for point in poly]))
        area = Polygon(poly).area
        person = Person(
            person_type=PersonType(label.lower()),
            location={"x": center_x, "y": center_y},
            area=int(area)
        )
        people.append(person)
    return people


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
