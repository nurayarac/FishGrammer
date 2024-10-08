!nvidia-smi
!pip install -q \
autodistill \
autodistill-grounded-sam \
autodistill-yolov8 \
supervision==0.9.0
import os
HOME=os.getcwd()
print(HOME)
!mkdir {HOME}/images
import cv2
import os

# Video dosyasının yolunu belirtin
VIDEO_FILE_PATH = "/content/denemee.mp4"  # Kendi video dosya yolunuzu buraya yazın
IMAGE_DIR_PATH = "/content/images"  # Çıkartılan görüntülerin kaydedileceği dizin yolunu buraya yazın
FRAME_STRIDE = 10  # Kaç karede bir görüntü çıkarılacağını belirtir

# Görüntülerin saklanacağı dizini oluşturun
if not os.path.exists(IMAGE_DIR_PATH):
    os.makedirs(IMAGE_DIR_PATH)

# Video dosyasını işleyin
video_name = os.path.splitext(os.path.basename(VIDEO_FILE_PATH))[0]
video_capture = cv2.VideoCapture(VIDEO_FILE_PATH)

frame_count = 0
image_count = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_count % FRAME_STRIDE == 0:
        image_file = f"{video_name}-frame-{image_count:05d}.jpg"
        image_path = os.path.join(IMAGE_DIR_PATH, image_file)
        cv2.imwrite(image_path, frame)
        image_count += 1

    frame_count += 1

video_capture.release()

print("Video başarıyla görüntülere dönüştürüldü.")
import supervision as sv

image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["png", "jpg", "jpg"])

print('image count:', len(image_paths))
IMAGE_DIR_PATH = f"{HOME}/images"
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 16)
import cv2
import supervision as sv

titles = [
    image_path.stem
    for image_path
    in image_paths[:SAMPLE_SIZE]]
images = [
    cv2.imread(str(image_path))
    for image_path
    in image_paths[:SAMPLE_SIZE]]

sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
!pip install roboflow
!pip install autodistill-grounded-sam


from autodistill.detection import CaptionOntology

ontology = CaptionOntology({
    "fish": "fish",
    "blue fish": "blue_fish",
    "large fish": "large_fish"
})

DATASET_DIR_PATH = f"{HOME}/dataset"
from autodistill_grounded_sam import GroundedSAM

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".jpg",
    output_folder=DATASET_DIR_PATH)
ANNOTATIONS_DIRECTORY_PATH ="/content/dataset/train/labels"
IMAGES_DIRECTORY_PATH = "/content/dataset/train/images"
DATA_YAML_PATH = "/content/dataset/data.yaml"
import supervision as sv

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH)

len(dataset)
import supervision as sv

# Tanımlamalar
SAMPLE_SIZE = 10  # Örneklemek istediğiniz görüntü sayısı
SAMPLE_GRID_SIZE = (2, 5)  # Görüntülerin düzenleneceği grid boyutu
SAMPLE_PLOT_SIZE = (20, 20)  # Her bir görüntünün boyutu

# Dataset'ten görüntü isimlerini alın
image_names = list(dataset.images.keys())[:SAMPLE_SIZE]

# BoxAnnotator oluşturun
box_annotator = sv.BoxAnnotator()

# Görüntüleri işleyin
images = []
for image_name in image_names:
    image = dataset.images[image_name]
    annotations = dataset.annotations[image_name]
    labels = [
        dataset.classes[class_id]
        for class_id in annotations.class_id]
    annotates_image = box_annotator.annotate(
        scene=image.copy(),
        detections=annotations,
        labels=labels)
    images.append(annotates_image)

# Görüntüleri grid şeklinde görselleştirin
sv.plot_images_grid(
    images=images,
    titles=image_names,
    grid_size=SAMPLE_GRID_SIZE,
    size=SAMPLE_PLOT_SIZE)
