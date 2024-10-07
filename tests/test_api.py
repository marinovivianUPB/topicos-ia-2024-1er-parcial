import enum
import pytest
import numpy as np
from src.predictor import GunDetector, annotate_detection, annotate_segmentation, match_gun_bbox
from src.models import Detection, Segmentation, PixelLocation, GunType, PersonType
from PIL import Image
from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture
def gun1():
    img_obj = Image.open("./gun1.jpg")
    img_array = np.array(img_obj)
    return img_array

@pytest.fixture
def gun2():
    img_obj = Image.open("./gun2.jpg")
    img_array = np.array(img_obj)
    return img_array

@pytest.fixture
def gun51():
    img_obj = Image.open("./51.png")
    img_array = np.array(img_obj)
    return img_array

def test_match_gun_bbox():

    bboxes = [[
      [
        181,
        340,
        415,
        495
      ]
    ],[
      [
        74,
        58,
        292,
        218
      ]
    ]]

    polygons = [[
        [241,89,1126,809]
    ],[
      [
        507,
        157,
        599,
        337
      ],
      [
        321,
        149,
        400,
        329
      ],
      [
        223,
        146,
        318,
        337
      ],
      [
        1,
        0,
        233,
        332
      ],
      [
        405,
        80,
        527,
        336
      ],
      [
        304,
        122,
        365,
        332
      ]
    ]]
    segments = []

    for polys in polygons:
        segment_list=[]
        for segment in polys:
            aux = [[segment[0],segment[1]],[segment[2],segment[1]],[segment[2],segment[3]],[segment[0],segment[3]]]
            segment_list.append(aux)
        segments.append(segment_list)

    print(segments)


    #polygons = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    matched_boxes = [[bboxes[0][0]], [None,None,bboxes[1][0],bboxes[1][0],None,None]]

    for bbox, segment_list, matched in zip(bboxes,segments,matched_boxes):
        for i,poly in enumerate(segment_list):
            matched_box = match_gun_bbox(poly, bbox)
            assert matched_box == matched[i]

def test_annotate_detection(gun1):
    detection = Detection(
        pred_type="OD",
        n_detections=1,
        boxes=[[
            181,
            340,
            415,
            495
        ]],
        labels=["pistol"],
        confidences=[0.83]
    )
    annotated_image = annotate_detection(gun1, detection)
    
    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)

def test_annotate_segmentation(gun1):
    segmentations = [
        [241,89,1126,809]
    ]

    segment_list=[]
    for segment in segmentations:
        aux = [[segment[0],segment[1]],[segment[2],segment[1]],[segment[2],segment[3]],[segment[0],segment[3]]]
        segment_list.append(aux)

    segmentation = Segmentation(
        pred_type="SEG",
        n_detections=1,
        polygons=segment_list,
        boxes=segmentations,
        labels=["danger"]
    )
    annotated_image = annotate_segmentation(gun1, segmentation, draw_boxes=True)

    assert annotated_image is not None
    assert isinstance(annotated_image, np.ndarray)

@pytest.mark.parametrize("threshold", [0.5, 0.8])
def test_detect_guns(gun51, threshold):
    detector = GunDetector()
    detection = detector.detect_guns(gun51, threshold=threshold)
    
    assert detection is not None
    assert isinstance(detection, Detection)
    assert detection.pred_type == "OD"
    assert detection.n_detections == 2

def test_segment_people(gun2):
    detector = GunDetector()
    segmentation = detector.segment_people(gun2)
    
    assert segmentation is not None
    assert isinstance(segmentation, Segmentation)
    assert segmentation.pred_type == "SEG"
    assert segmentation.n_detections == 6

def test_get_info():
    data = {
        "model_name": "Gun detector",
        "gun_detector_model": "DetectionModel",
        "semantic_segmentation_model": "SegmentationModel",
        "input_type": "image"
    }
    response = client.get("/model_info")
    assert response.status_code == 200
    assert response.json() == data

def test_detect_guns():
    data = {
  "pred_type": "OD",
  "n_detections": 2,
  "boxes": [
    [
      162,
      102,
      325,
      203
    ],
    [
      9,
      73,
      103,
      127
    ]
  ],
  "labels": [
    "Rifle",
    "Rifle"
  ],
  "confidences": [
    0.8702183365821838,
    0.8644607663154602
  ]
}
    filename = "./51.png"
    response = client.post("/detect_guns", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == data

def test_annotate_guns():
    filename = "./51.png"
    response = client.post("/annotate_guns", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    assert response.headers["Content-Type"] in ["image/jpeg", "image/png"]
    assert response.status_code == 200

def test_detect_people():
    filename = "./gun2.jpg"
    response = client.post("/detect_people", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    response_dict = response.json()
    assert response.headers["Content-Type"] == "application/json"
    assert response_dict["n_detections"] == 6
    assert response_dict is not None
    assert response.status_code == 200

def test_annotate_people():
    filename = "./gun2.jpg"
    response = client.post("/annotate_people", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    assert response.headers["Content-Type"] in ["image/jpeg", "image/png"]
    assert response.status_code == 200

def test_detect():
    filename = "./gun2.jpg"
    response = client.post("/detect", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    assert response.headers["Content-Type"] == "application/json"
    assert response.json() is not None
    assert response.status_code == 200

def test_annotate():
    filename = "./gun2.jpg"
    response = client.post("/annotate", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    assert response.headers["Content-Type"] in ["image/jpeg", "image/png"]
    assert response.status_code == 200

def guns():
    data = [
  {
    "gun_type": "rifle",
    "location": {
      "x": 183,
      "y": 138
    }
  }
]
    filename = "./gun2.jpg"
    response = client.post("/guns", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    assert response.status_code == 200
    assert response.json() in data

def people():
    data = [
  {
    "person_type": "danger",
    "location": {
      "x": 683,
      "y": 441
    },
    "area": 354360
  }
]
    filename = "./gun1.jpg"
    response = client.post("/people", files={"file": ("filename", open(filename, "rb"), "image/jpeg")})
    assert response.status_code == 200
    assert response.json() in data