# Step: Stitch the images together based on the ArUco markers
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


import argparse

VIDEO_DIR = "C:/Users/kklym/Documents/GitHub/vertical_distance_mapping_warehouse/sample.mp4"
IMAGE_DIR = "C:/Users/kklym/Documents/GitHub/vertical_distance_mapping_warehouse/frames"
IMG_SCALE = 4900

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

    for filename in sorted(os.listdir(stitched_dir)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(stitched_dir, filename)
            print("[INFO] loading image...")
            image = cv2.imread(image_path)

            image = cv2.resize(image, (IMG_SCALE, int(image.shape[0] * (IMG_SCALE / image.shape[1]))))  # used sample width based on stitching 5 frames per pano

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


                    # Record the data
                    with open(data_file_path, 'a') as data_file:
                        data_file.write(f"{filename},{markerID},{topLeft},{topRight},{bottomRight},{bottomLeft},{center}\n")

                    # Draw the bounding box and ID on the image
                    cv2.polylines(image, [np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



                    # Crazy binary choice soritng ahah
                    # 1 el: positioned as right
                    # 2 el: one is left another right
                    # 3 el there is a l, m, r

                    if aruco_markers.get(filename + "_r") is None or center[0] > aruco_markers[filename + "_r"][1][0]:
                        if aruco_markers[filename + "_r"]:
                            if aruco_markers[filename + "_l"] and center[0] > aruco_markers[filename + "_l"][1][0]:
                                aruco_markers[filename + "_m"] = [markerID, center]
                                break
                            else:
                                aruco_markers[filename + "_l"] = aruco_markers[filename + "_r"]
                        aruco_markers[filename + "_r"] = [markerID, center]


                    elif aruco_markers.get(filename + "_l") is None or center[0] < aruco_markers[filename + "_l"][1][0]:
                        if aruco_markers[filename + "_l"]:
                            if aruco_markers[filename + "_r"] and center[0] < aruco_markers[filename + "_r"][1][0]:
                                aruco_markers[filename + "_m"] = [markerID, center]
                                break
                            else:
                                aruco_markers[filename + "_r"] = aruco_markers[filename + "_l"]
                        aruco_markers[filename + "_l"] = [markerID, center]


                    else:
                        aruco_markers[filename + "_m"] = [markerID, center]
                        break

                # Save the image with detected markers
                #cv2.imwrite(os.path.join(IMAGE_DIR, "stitched_aruco", f"{filename}"), image)
            else:
                print(f"No markers detected in {filename}.")
    return aruco_markers


def stitch_frames_right_movement(aruco_markers, stitched_dir):
    sorted_files = sorted(os.listdir(stitched_dir))

    def get_next_file(current_filename):
        try:
            current_index = sorted_files.index(current_filename)
            if filename.endswith(".jpg"): 
                return sorted_files[current_index + 1]  # Get the next file in the list
            else:
                return None
        except (ValueError, IndexError):
            return None
        
    for filename in sorted_files:
        if filename.endswith(".jpg"):

            # Open Image 1
            image1_path = os.path.join(stitched_dir, filename)
            print("[INFO] loading image...")
            image1 = cv2.imread(image1_path)
            image1 = cv2.resize(image1, (IMG_SCALE, int(image1.shape[0] * (IMG_SCALE / image1.shape[1]))))  # used sample width based on stitching 5 frames per pano


            # Open Image 2
            filename2 = get_next_file(filename)
            if filename2 is None:
                continue
            image2_path = os.path.join(stitched_dir, filename2)
            image2 = cv2.imread(image2_path)
            image2 = cv2.resize(image2, (IMG_SCALE, int(image2.shape[0] * (IMG_SCALE / image2.shape[1]))))  # used sample width based on stitching 5 frames per pano


            print(f"{filename} - {filename2}")
            
            # Get ArUco marker coordinates for alignment
            #mid1 = aruco_markers.get(filename + "_r")
            #mid2 = aruco_markers.get(filename2 + "_l")


            # MATCH tags from 2 consequent images
            ext = ["_r", "_m", "_l"]
            found = False
            for e1 in ext:
                if not found:
                    mid1 = aruco_markers.get(filename + e1)   
                    for e2 in reversed(ext):
                        mid2 = aruco_markers.get(filename2 + e2)
                        if mid1 and mid2 and mid1[0] == mid2[0]:
                            found = True
                            break
                            

            # Create a blank canvas
            height = max(image1.shape[0], image2.shape[0])
            width = image1.shape[1] + image2.shape[1]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)


            move_x = IMG_SCALE - mid1[1][0] + mid2[1][0]
            move_y = mid1[1][1] - mid2[1][1]  # Assuming mid1 and mid2 contain y-coordinates as well

            canvas_width = image1.shape[1] + image2.shape[1] - abs(move_x)
            canvas_height = max(image1.shape[0], image2.shape[0]) + abs(move_y)
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Place image1 on the canvas
            start_y_image1 = max(0, move_y) if move_y < 0 else 0
            canvas[start_y_image1:start_y_image1 + image1.shape[0], :image1.shape[1]] = image1

            # Calculate the starting x and y coordinates for image2 based on the move
            start_x_image2 = max(0, image1.shape[1] - abs(move_x))
            start_y_image2 = max(0, -move_y) if move_y > 0 else 0
            canvas[start_y_image2:start_y_image2 + image2.shape[0], start_x_image2:start_x_image2 + image2.shape[1]] = image2

            # Blend both images onto the canvas with alpha 0.5 and increase brightness
            alpha = 0.5
            beta = 50  # Increase brightness by adding a constant value

            # Blend image1 onto the canvas
            overlay1 = cv2.addWeighted(canvas[start_y_image1:start_y_image1 + image1.shape[0], :image1.shape[1]], 1 - alpha, image1, alpha, 0)
            canvas[start_y_image1:start_y_image1 + image1.shape[0], :image1.shape[1]] = cv2.convertScaleAbs(overlay1, alpha=1.0, beta=beta)

            # Blend image2 onto the canvas
            overlay2 = cv2.addWeighted(canvas[start_y_image2:start_y_image2 + image2.shape[0], start_x_image2:start_x_image2 + image2.shape[1]], 1 - alpha, image2, alpha, 0)
            canvas[start_y_image2:start_y_image2 + image2.shape[0], start_x_image2:start_x_image2 + image2.shape[1]] = cv2.convertScaleAbs(overlay2, alpha=1.0, beta=beta)

            print(f"{mid1} - {mid2}")

            plt.imshow(canvas)
            plt.show()



if __name__ == "__main__":
    aruco_markers = detect_aruco()

    print(aruco_markers)
    stitch_frames_right_movement(aruco_markers, os.path.join(IMAGE_DIR, "stitched_aruco"))
