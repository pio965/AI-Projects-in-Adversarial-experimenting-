# AI-Projects-in-Adversarial-experimenting-
# 🚗 Adversarial Attacks on KITTI-Based Perception Models

This project demonstrates how to simulate adversarial attacks on a pretrained ResNet-18 classifier using the KITTI dataset and train a deep learning model to detect such attacks. It integrates LiDAR visualization, FGSM attack generation, and adversarial detection using PyTorch.

---

## 🔍 Project Structure

- **Dataset**: KITTI Object Detection dataset (minimal subset with camera images + LiDAR).
- **Attack Method**: Fast Gradient Sign Method (FGSM).
- **Model**: ResNet-18 for classification + custom CNN for adversarial detection.
- **Tools**: PyTorch, OpenCV, Open3D, Matplotlib, torchvision, pykitti.

---

🧪 Key Features

- Visualizes 3D LiDAR point clouds.
- Applies adversarial noise to KITTI camera images.
- Trains a CNN to classify clean vs attacked images.
- Evaluates the accuracy of adversarial detection.

---

 🚀 How to Run
 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/adversarial-kitti-experiments.git
cd adversarial-kitti-experiments


2. Install Dependencies
Use a virtual environment or Google Colab:

Locally (Linux/macOS):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python open3d matplotlib tqdm pykitti numpy
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt && cd ..

Download KITTI Dataset (Subset)
bash
Copy code
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
unzip data_object_image_2.zip -d ./kitti_data
unzip data_object_velodyne.zip -d ./kitti_data


4. Run the Script
python adversarial_experiments.py


