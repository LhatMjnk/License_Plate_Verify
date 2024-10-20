# License Plate Detection and Recognition

This project focuses on detecting and recognizing license plates using YOLOv11 models. The workspace is organized into several directories for data, models, and utilities.

## Project Structure
```
__init__.py
.dockerignore
.gitignore
App/
    app.py
compose.yaml
Data/ # Need to be download
    Detection/
        data.yaml
        images/
            train/
            val/
        labels/
            train/
            val/
    OCR/
        data.yaml
        images/
        labels/
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
```

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/LhatMjnk/License_Plate_Verify.git
    cd License_Plate_Verify
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the application:**
    ```sh
    streamlit App/app.py
    ```

## Data
Need to be downloaded at: [Data](https://drive.google.com/drive/folders/1OZnFA6JCeAE4eX7U6wXTecnWLpfeZpj-?usp=sharing)
- **Detection Data:** Located in `Data/Detection/`, contains images and labels for training and validation.
- **OCR Data:** Located in `Data/OCR/`, contains images, labels, and datasets for OCR training.

## App
After running the app with ```streamlit``` you will get the interface:
![App interface](https://github.com/user-attachments/assets/a455cd77-678a-42da-9c2d-e5d48c7faecf)

The database I used in this project is PostgreSQL. 
You must fill in all the database boxes (the port is optional) and the app is ready to use.
![App interface when connect to database successfully](https://github.com/user-attachments/assets/7ab14a9e-d0d3-47d0-9f5e-1d891c140e93)


**Caution:** The URL when using the camera in the [App/app.py](App/app.py) is getting by installing "IP Webcam" and setting the port to 8080
**Caution 2:** The test_vid.MOV need to be downloaded in this link [Test_video](https://drive.google.com/file/d/1qg0qradfjQ5j9Adb0J2Zm2C6zH-ZKBHC/view?usp=sharing) and move to folder [App/](App/)


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
- **To use:** Clone this repository and running:
    ```sh
    docker-compose up
    ```

## Acknowledgements

- YOLOv11 models
- Ultralytics

## Contact

For any inquiries, please contact [21013299@st.phenikaa-uni.edu.vn](mailto:21013299@st.phenikaa-uni.edu.vn).
