# VisionDrive: Deep Learning for Enhanced Autonomous Driving

A deep learning-based system designed to improve autonomous driving by performing critical tasks such as lane detection, wheel angle prediction, traffic sign classification, and vehicle detection.

## About

The **VisionDrive** project aims to enhance the reliability and safety of autonomous navigation through deep learning models. The system is designed to tackle key challenges in self-driving technology, including environmental perception, lane detection, and real-time decision-making.

## Data

The project utilizes multiple datasets to train different models for various autonomous driving tasks:

- **NVIDIA dataset** – Used for training a CNN-based model to predict steering angles based on detected lanes and surrounding traffic.
- **TUSimple dataset** – Used for lane detection with **DeepLabV3+** for high-precision semantic segmentation.
- **LISA Traffic Light Dataset** – Used to fine-tune a **YOLOv3** model for real-time traffic sign classification.
- **NVIDIA dataset for Vehicle Detection** – Used with a pre-trained **YOLOv8 Nano** model to detect vehicles and pedestrians accurately.

## Features

- **Vehicle Detection** using YOLOv8 Nano for efficient identification of vehicles and pedestrians.
- **Traffic Light Classification** using a fine-tuned YOLOv3 model for real-time detection of traffic signals.
- **Lane Detection** via DeepLabV3+ semantic segmentation to generate lane overlays.
- **Wheel Angle Prediction** using a CNN model trained on the NVIDIA dataset to determine appropriate steering angles.

## Contributing

Contributions are welcome! If you have ideas to enhance the system’s accuracy, expand dataset coverage, or improve real-time performance, feel free to fork the repo and submit a pull request.

## Support

If you find this project helpful, consider giving it a ⭐ on GitHub!

## Acknowledgments

- NVIDIA Deep Learning Research
- LISA Traffic Light Dataset
- TUSimple Lane Detection Dataset
- YOLO Object Detection Framework
- DeepLabV3+ for Semantic Segmentation
