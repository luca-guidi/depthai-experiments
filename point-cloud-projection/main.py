#!/usr/bin/env python3
import json
import os
import tempfile
import platform
from pathlib import Path
import ransac

import cv2
import depthai
import numpy


try:
    from projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e} \033[0m ")


device = depthai.Device("", False)
pipeline = device.create_pipeline(config={
    'streams': ['right', 'depth'],
    'ai': {
        "blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
    },
    'camera': {'mono': {'resolution_h': 720, 'fps': 30}},
})

if pipeline is None:
    raise RuntimeError("Error creating a pipeline!")

right = None
pcl_converter = None

while True:
    data_packets = pipeline.get_available_data_packets()

    for packet in data_packets:
        if packet.stream_name == "right":
            right = packet.getData()
            #cv2.imshow(packet.stream_name, right)
        elif packet.stream_name == "depth":
            frame = packet.getData()
            median = cv2.medianBlur(frame, 5)
            median2 = cv2.medianBlur(median,5)
            '''
            median3 = cv2.medianBlur(median,5)
            median4 = cv2.medianBlur(median,5)
            median5 = cv2.medianBlur(median,5)

            bilateral = cv2.bilateralFilter(frame,15,75,75)
            '''
            if right is not None:

                if pcl_converter is None:
                    fd, path = tempfile.mkstemp(suffix='.json')
                    with os.fdopen(fd, 'w') as tmp:
                        json.dump({
                            "width": 1280,
                            "height": 720,
                            "intrinsic_matrix": [item for row in device.get_right_intrinsic() for item in row]
                        }, tmp)

                    pcl_converter = PointCloudVisualizer(path)
                pcd = pcl_converter.rgbd_to_projection(median, right)
                pcl_converter.visualize_pcd()
                print("**************")
                print(numpy.asarray((pcl_converter.pcl.points)))

                # x,y,z = ransac.find_plane(pcd)
                # ransac.show_graph(x,y,z)
            #cv2.imshow(packet.stream_name, frame)
            '''
            cv2.imshow("filter", median)

            cv2.imshow("filter2", median2)
            cv2.imshow("filter3", median3)
            cv2.imshow("filter4", median4)
            '''
            #cv2.imshow("filter5", median5)


            #cv2.imshow("filter2", bilateral)

    if cv2.waitKey(1) == ord("q"):
        break

if pcl_converter is not None:
    pcl_converter.close_window()
