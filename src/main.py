import stitch
import measure_aruco
import segm_rotate
import detect_poles
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "results")
FRAME_DIR = os.path.join(IMAGE_DIR, "sorting", "raw")
CLAS_DIR = os.path.join(IMAGE_DIR, "sorting", "detected")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
VIDEO_DIR = os.path.join(BASE_DIR, "data/raw", "sample1.mp4")
NEW_VID_DIR = os.path.join(BASE_DIR, "data/raw", "sample1_rev.mp4")


POLES_CONF = 0.6
FRAMES_PER_PANO = 6 # works only on even numbers
FRAME_RATE = 9

def plot_stats(dist):
    #testing
    ave_list = []
    counter = 0
    prev = 0
    for key in dist:
        counter += 1
        if counter % 2 == 1:
            prev = dist.get(key)
        else:
            ave_list.append(dist.get(key) + prev)

    # Create the distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(ave_list, kde=True, bins=100)

    # Customize the plot
    plt.title('Distribution of Data')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Basic statistics:
    print("Basic statistics:")
    print(f"Mean: {np.mean(ave_list)}")
    print(f"Median: {np.median(ave_list)}")
    print(f"Standard Deviation: {np.std(ave_list)}")

    print("-----------------------------------------------------------------")
    print(f"The MEDIAN for the in-between pole distances: {np.median(ave_list)} taken from {counter // 2} sois")

    # Show the plot
    plt.show()

if __name__ == "__main__":  
     
    #video = stitch.reverse(VIDEO_DIR, VIDEO_DIR)
    stitch.test_process_video_vertically(VIDEO_DIR, NEW_VID_DIR)
    VIDEO_DIR = NEW_VID_DIR
    
    # Normal Image Extraction
    #stitch.extract_frames(VIDEO_DIR, IMAGE_DIR, frame_rate = 9, frames_per_pano = 6)
    
    # Image Extraction using poles detection
    #detect_poles.extract_frames(SOURCE_DIR = VIDEO_DIR, FRAME_DIR = FRAME_DIR, FRAME_RATE = FRAME_RATE)
    #poles = detect_poles.detect_poles(FRAME_DIR = FRAME_DIR, OUTPUT_DIR = CLAS_DIR, WEIGHTS_DIR = WEIGHTS_DIR, POLES_CONF = POLES_CONF)
    #detect_poles.sort_images(SOURCE_DIR= FRAME_DIR, DEST_DIR = IMAGE_DIR, FRAMES_PER_PANO = FRAMES_PER_PANO, poles=poles)
    
    #input("Look at first folders before drone starts moving and delete those")

    #stitch.set_stitch(IMAGE_DIR)
    #segm_rotate.rotate_yolo(BASE_DIR=BASE_DIR, IMAGE_DIR=IMAGE_DIR)
    
    #stitch.mark_aruco(SOURCE_DIR = os.path.join(IMAGE_DIR, "rotated"), IMAGE_DIR = IMAGE_DIR)
    
    #input("Stitching and rotating is DONE - Proceed? [Y/n]")

    
     
    #aruco_markers, aruco_width = measure_aruco.detect_aruco(IMAGE_DIR=IMAGE_DIR)
    #print(aruco_markers, aruco_width)
    
    #dist = measure_aruco.relative_distance(aruco_markers, aruco_width, IMAGE_DIR)

    #plot_stats(dist)

