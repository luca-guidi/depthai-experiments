#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
from pathlib import Path

'''
FastDepth demo running on device.
https://github.com/zl548/MegaDepth


Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Onnx taken from PINTO0309, added scaling flag, and exported to blob:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/153_MegaDepth
'''

# --------------- Arguments ---------------
nn_path = "models/megadepth_192x256_openvino_2021.4_6shave.blob"
curr_path = Path(__file__).parent.resolve()

# choose width and height based on model
NN_WIDTH, NN_HEIGHT = 256, 192

# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Define a neural network
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Define camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(NN_WIDTH, NN_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create outputs
xout_cam = pipeline.createXLinkOut()
xout_cam.setStreamName("cam")

xout_vid = pipeline.createXLinkOut()
xout_vid.setStreamName("video")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

# Link
cam.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_cam.input)
detection_nn.out.link(xout_nn.input)
cam.video.link(xout_vid.input)


# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_vid = device.getOutputQueue("video", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    count = 0
    while True:
        in_frame = q_cam.get()
        in_nn = q_nn.get()
        in_vid = q_vid.get()

        frame = in_frame.getCvFrame()
        rgb = in_vid.getCvFrame()
        dest_width = rgb.shape[0] * NN_WIDTH / NN_HEIGHT
        offset = int((rgb.shape[1] - dest_width) // 2)
        rgb = rgb[:, offset : int(dest_width) + offset]
        # rgb = 
        # Get output layer
        pred = np.array(in_nn.getFirstLayerFp16()).reshape((NN_HEIGHT, NN_WIDTH))

        # Scale depth to get relative depth
        d_min = np.min(pred)
        d_max = np.max(pred)
        depth_relative = (pred - d_min) / (d_max - d_min)

        # Color it
        depth_relative = np.array(depth_relative) * 255
        depth_relative = depth_relative.astype(np.uint8)
        depth_relative = 255 - depth_relative
        depth_relative_black = depth_relative.copy()
        depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)

        # Show FPS
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        # Concatenate NN input and produced depth
        cv2.imshow("Detections", cv2.hconcat([frame, depth_relative]))
        depth_relative2 = cv2.resize(depth_relative_black, (rgb.shape[1], rgb.shape[0]), cv2.INTER_CUBIC)
        depth_relative2 = cv2.cvtColor(depth_relative2, cv2.COLOR_GRAY2BGR)
        concated_rgbd = cv2.hconcat([rgb, depth_relative2])
        cv2.imshow("Detections full size", concated_rgbd)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            rgbd_folder = str(curr_path) + '/pcl_dataset/rgb_depth/'
            # pcl_converter.save_ply(ply_pth)
            # pcl_converter.save_mesh_from_rgbd(ply_pth)
            count += 1
            filename = rgbd_folder + 'rgbd_' + str(count) + '.png'

            cv2.imwrite(filename, concated_rgbd)
            print("Saving")