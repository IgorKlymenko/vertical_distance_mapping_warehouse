import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


import argparse
import imutils

VIDEO_DIR = "C:/Users/kklym/Documents/GitHub/IKK/sample.mp4"
IMAGE_DIR = "C:/Users/kklym/Documents/GitHub/IKK/frames"


ap = argparse.ArgumentParser()
# Remove argparse related code
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image containing ArUCo tag")
# ap.add_argument("-t", "--type", type=str,
# 	default="DICT_ARUCO_ORIGINAL",
# 	help="type of ArUCo tag to detect")
# args = vars(ap.parse_args())




ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}



def extract_frames(video_path, output_folder, frame_rate, frames_per_pano):
    frames_per_stitch = frame_rate * frames_per_pano   #5 frames per panorama -- empiracally measured
    all_frames = 600

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_frame_count = 0
    for i in range(0, all_frames, frames_per_stitch):
        curr_dir = output_folder + f"/set_{i//frames_per_stitch}"
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        while cap.isOpened() and frame_count < i + frames_per_stitch:

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_rate == 0:
                # Save the frame as an image file
                frame_filename = os.path.join(curr_dir, f'frame_{saved_frame_count:04d}.jpg')
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

            frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames to '{output_folder}'")

def img_stitch(img_dir):  
    stitcher = cv2.Stitcher_create()

    # Extract all frames from the directory
    frames = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(img_dir, filename)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {filename}")

    # Edge case
    if len(frames) == 0:
        print("Error: No frames to stitch.")
        return

    status, stitched_image = stitcher.stitch(frames)

    # Debugging
    if status != cv2.Stitcher_OK:
        print("Error during stitching. Status code:", status)
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Error: Need more images to perform stitching.")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Error: Homography estimation failed.")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Error: Camera parameters adjustment failed.")
        else:
            print("Error: Unknown error occurred during stitching.")

    return stitched_image

def set_stitch():
    for filename in sorted(os.listdir(IMAGE_DIR)):
        stitched_image = img_stitch(os.path.join(IMAGE_DIR, filename))
        
        if not os.path.exists(os.path.join(IMAGE_DIR, "stitched")):
            os.makedirs(os.path.join(IMAGE_DIR, "stitched"))

        cv2.imwrite(os.path.join(IMAGE_DIR, "stitched", f"{filename}.jpg"), stitched_image)
        if stitched_image is None:
            print(f"Skipping {filename} due to stitching error.")

def detect_aruco():
    if not os.path.exists(os.path.join(IMAGE_DIR, "stitched_aruco")):
        os.makedirs(os.path.join(IMAGE_DIR, "stitched_aruco"))


    stitched_dir = os.path.join(IMAGE_DIR, "stitched")
    if not os.path.exists(stitched_dir):
        print(f"Error: Directory {stitched_dir} does not exist.")
        return

    for filename in sorted(os.listdir(stitched_dir)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(stitched_dir, filename)
            print("[INFO] loading image...")
            image = cv2.imread(image_path)
            image = imutils.resize(image, width=4900) #used sample width based on stitching 5 frames per pano


            args = {
                "image": str(image_path),  # Set your image path here
                "type": "DICT_4X4_1000"       # Set the ArUCo tag type here
            }

            
        if ARUCO_DICT.get(args["type"], None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(
                args["type"]))
            sys.exit(0)

        # load the ArUCo dictionary, grab the ArUCo parameters, and detect
        # the markers
        print("[INFO] detecting '{}' tags...".format(args["type"]))
        arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        (corners, ids, rejected) = detector.detectMarkers(image)
        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[INFO] ArUco marker ID: {}".format(markerID))

            # upload the output image
            cv2.imwrite(os.path.join(IMAGE_DIR, "stitched_aruco", f"{filename}.jpg"), image)
            if image is None:
                print(f"Skipping {filename} due to stitching error.")


if __name__ == "__main__":
    extract_frames(VIDEO_DIR, IMAGE_DIR, 10, 6)
    set_stitch()
    detect_aruco()
