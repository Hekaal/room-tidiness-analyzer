# detector/classes.py

# Mapping class_id YOLOv8 (COCO) → nama kelas
COCO_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    39: "bottle",
    41: "cup",
    56: "chair",
    57: "couch",
    59: "bed",
    73: "laptop",
    84: "book",
}

# Kelas yang relevan untuk analisis kerapian
ALLOWED_CLASSES = {
    "backpack", "handbag", "bottle", "cup",
    "book", "laptop",
    "chair", "bed", "couch",
    "person"
}
