# Warehouse Distances Measurement and Vertical Stitching Pipeline Using Drone Imagery

Custom image processing pipeline for warehouse distance mapping and object detection designed for Fling Ltd

## Features

- Video frame extraction
- Panoramic image stitching for specific types of location
- ArUco marker and it's orientation detection
- Object segmentation (YOLO + SAM) using pretrained model
- Image rotation correction based on segmented elements
- Multipole distance calculation and analysis

## Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download model weights:
   - SAM: `sam_vit_h_4b8939.pth`
   - YOLO: `best.pt`
   Place in `weights` directory

## Usage

Put the video for procewsing into data/raw folder in your code dir
Run main script: `python main.py`

## Structure

- `main.py`: Main Execution Script
- `stitch.py`: Frame Extraction and Stitching
- `measure_aruco.py`: ArUco Marker Processing
- `segm_rotate.py`: Segmentation and Rotation

## Notes

- GPU recommended for processing
- titching and ArUco detection parameters adjustable for environment
- Adaptable for various industrial mapping applications
- All folders will be created automatically on your machine (this repo has folder results which includes one sample of stitched set_10).
  
Developed for warehouse inventory and layout analysis. Customizable for various industrial applications.
