# License Plate Detection and Recognition

This project focuses on detecting and recognizing license plates using YOLOv11 models. The workspace is organized into several directories for data, models, and utilities.

## Project Structure

__init__.py
.dockerignore
.gitignore
App/
    app.py
compose.yaml
Data/
    Detection/
        data.yaml
        images/
            train/
            train.cache
            val/
        labels/
            train/
            train.cache
            val/
            val.cache
    OCR/
        data.yaml
        dataset/
            0.txt
            ...
        images/
            ...
        labels/
        Licens plates.v13i.yolov11/
Dockerfile
Model/
    LP_Detect_YOLOv11n.pt
    LP_Detection.ipynb
    LP_Recog_YOLOv11n.pt
    LP_Recognizer.ipynb
    runs/
        detect/
    yolo11n.pt
README.md
requirements.txt
utils/
    __init__.py
    __pycache__/
    utils.py

## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the application:**
    ```sh
    python App/app.py
    ```

## Data

- **Detection Data:** Located in `Data/Detection/`, contains images and labels for training and validation.
- **OCR Data:** Located in `Data/OCR/`, contains images, labels, and datasets for OCR training.

## Models

- **License Plate Detection:** YOLOv11 model for detecting license plates, stored in `Model/LP_Detect_YOLOv11n.pt`.
- **License Plate Recognition:** YOLOv11 model for recognizing license plates, stored in `Model/LP_Recog_YOLOv11n.pt`.

## Notebooks

- **Detection Notebook:** [Model/LP_Detection.ipynb](Model/LP_Detection.ipynb)
- **Recognition Notebook:** [Model/LP_Recognizer.ipynb](Model/LP_Recognizer.ipynb)

## Utilities

- **Utility Functions:** Located in `utils/utils.py`, includes functions like `organize_tex_label_folder`.

## Docker

- **Dockerfile:** Used to build the Docker image.
- **Compose File:** [compose.yaml](compose.yaml) for setting up the Docker environment.

## Acknowledgements

- YOLOv11 models
- Ultralytics

## Contact

For any inquiries, please contact [21013299@st.phenikaa-uni.edu.vn](mailto:21013299@st.phenikaa-uni.edu.vn).