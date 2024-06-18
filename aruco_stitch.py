# Step: Stitch the images together based on the ArUco markers
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


import argparse

VIDEO_DIR = "C:/Users/kklym/Documents/GitHub/vertical_distance_mapping_warehouse/sample.mp4"
IMAGE_DIR = "C:/Users/kklym/Documents/GitHub/vertical_distance_mapping_warehouse/frames"


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


def detect_aruco():
    if not os.path.exists(os.path.join(IMAGE_DIR, "stitched_aruco")):
        os.makedirs(os.path.join(IMAGE_DIR, "stitched_aruco"))

    stitched_dir = os.path.join(IMAGE_DIR, "stitched").replace("\\", "/")
    if not os.path.exists(stitched_dir):
        print(f"Error: Directory {stitched_dir} does not exist.")
        return
    # Prepare a file to record detected ArUco markers data
    data_file_path = os.path.join(IMAGE_DIR, "stitched_aruco", "detected_aruco_data.csv")
    with open(data_file_path, 'w') as data_file:
        data_file.write("Image,MarkerID,TopLeft,TopRight,BottomRight,BottomLeft,Center\n")

    # Dictionary to store the first and last detected ArUco markers for each image
    aruco_markers = {}
    aruco_markers_lt = {}
    aruco_markers_m = {}

    for filename in sorted(os.listdir(stitched_dir)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(stitched_dir, filename)
            print("[INFO] loading image...")
            image = cv2.imread(image_path)
            image = cv2.resize(image, (4900, int(image.shape[0] * (4900 / image.shape[1]))))  # used sample width based on stitching 5 frames per pano

            aruco_type = "DICT_4X4_1000"
            if ARUCO_DICT.get(aruco_type, None) is None:
                print("[INFO] ArUco tag of '{}' is not supported".format(aruco_type))
                sys.exit(0)

            # load the ArUco dictionary, grab the ArUco parameters, and detect the markers
            print("[INFO] detecting '{}' tags...".format(aruco_type))
            arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
            arucoParams = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
            corners, ids, rejected = detector.detectMarkers(image)

            # verify *at least* one ArUco marker was detected
            if ids is not None:
                ids = ids.flatten()

                # Initialize dictionary entries for the first and last detected markers
                aruco_markers[filename + "_l"] = None
                aruco_markers[filename + "_r"] = None

                # loop over the detected ArUco corners
                for i, (markerCorner, markerID) in enumerate(zip(corners, ids)):
                    # extract the marker corners
                    corners = markerCorner.reshape((4, 2))
                    topLeft, topRight, bottomRight, bottomLeft = corners
                    topLeft = tuple(map(int, topLeft))
                    topRight = tuple(map(int, topRight))
                    bottomRight = tuple(map(int, bottomRight))
                    bottomLeft = tuple(map(int, bottomLeft))

                    # compute the center coordinates of the ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    center = (cX, cY)

                    aruco_markers_lt[markerID] = topLeft
                    aruco_markers_m[markerID] = center

                    # Record the data
                    with open(data_file_path, 'a') as data_file:
                        data_file.write(f"{filename},{markerID},{topLeft},{topRight},{bottomRight},{bottomLeft},{center}\n")

                    # Draw the bounding box and ID on the image
                    cv2.polylines(image, [np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Update dictionary with the first, second, and last detected markers
                    if aruco_markers.get(filename + "_l") is None:
                        aruco_markers[filename + "_l"] = markerID
                    elif aruco_markers.get(filename + "_r") is None:
                        aruco_markers[filename + "_r"] = markerID
                    aruco_markers[filename + "_m"] = markerID
                # Save the image with detected markers
                #cv2.imwrite(os.path.join(IMAGE_DIR, "stitched_aruco", f"{filename}"), image)
            else:
                print(f"No markers detected in {filename}.")

    # Print the dictionary containing the first and last detected markers
    print(aruco_markers_lt)
    print(aruco_markers_m)

    return aruco_markers, aruco_markers_lt, aruco_markers_m



def stitch_frames(aruco_markers, aruco_markers_lt, aruco_markers_m):
    for idx, key in enumerate(sorted(aruco_markers.keys())[:-1]):  # Exclude the last key to avoid index out of range
        current_key_str = os.path.join(IMAGE_DIR, "stitched_aruco", key[:-2])
        curr = idx + 1
        next_key = sorted(aruco_markers.keys())[curr]
        
        while aruco_markers[key] != aruco_markers[next_key] and curr < len(aruco_markers.keys()) - 1:
            curr += 1
            next_key = sorted(aruco_markers.keys())[curr]

        next_key_str = os.path.join(IMAGE_DIR, "stitched_aruco", next_key[:-2])

        img1 = cv2.imread(current_key_str)
        img2 = cv2.imread(next_key_str)

        if img1 is None or img2 is None:
            continue

        # Get ArUco marker coordinates for alignment
        mid1 = aruco_markers_m.get(aruco_markers[key])
        mid2 = aruco_markers_m.get(aruco_markers[next_key])

        if None in [mid1, mid2]:
            continue

        # Calculate translation vectors using centers
        translation_x = mid2[0] - mid1[0]
        translation_y = mid2[1] - mid1[1]

        # Create translation matrix
        translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        img1_translated = cv2.warpAffine(img1, translation_matrix, (img1.shape[1], img1.shape[0]))

        # Create a new image large enough to hold both translated img1 and img2
        new_height = max(img1_translated.shape[0], img2.shape[0])
        new_width = img1_translated.shape[1] + img2.shape[1] - abs(translation_x)  # Adjust width to overlap on the ArUco tag
        combined_image = np.zeros((new_height, new_width, 3), dtype=np.float32)
        
        # Calculate the alpha blending factor
        alpha = 0.5

        # Place img1_translated and img2 in the new image with proper overlap
        overlap_start = img1_translated.shape[1] - abs(int(translation_x))  # Start of overlap
        combined_image[:img1_translated.shape[0], :overlap_start] = img1_translated[:, :overlap_start] * alpha
        combined_image[:img2.shape[0], overlap_start:overlap_start + img2.shape[1]] = img2 * alpha

        # Blend overlapping areas
        combined_image[:img1_translated.shape[0], overlap_start:overlap_start + img2.shape[1]] += img1_translated[:, overlap_start - img2.shape[1]:overlap_start] * alpha

        # Convert the combined_image back to uint8 type after processing
        combined_image = combined_image.astype(np.uint8)
        # Display using matplotlib
        plt.figure(figsize=(20, 20))
        plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Combined Image: {key} with {next_key}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    aruco_markers, aruco_markers_lt, aruco_markers_m = detect_aruco()
    stitch_frames(aruco_markers, aruco_markers_lt, aruco_markers_m)
