# FCN-plus-Depth
Capture color + depth from Intel RealSense
Find a person using a Fully Convolutional Neural Net
Find the distance to that person using the RealSense depth frame
Stream that to a browser using the Flask framework

To move to a Jetson TX1, need to update some of the pylibrealsense calls (colout -> color, wait_for_frame -> wait_for_frames)
Also need to fix the find_contours return value (remove the leading underscore - only 2 values in that OpenCV version)
And confusingly, need to do export DISPLAY=0: before running it.
