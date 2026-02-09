import pybullet as p
import numpy as np
import cv2
from config import *


class VisionSystem:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        # Matrix setup for PyBullet Camera
        self.aspect = CAMERA_WIDTH / CAMERA_HEIGHT
        self.proj_matrix = p.computeProjectionMatrixFOV(CAMERA_FOV, self.aspect, 0.1, 100.0)

    def get_camera_image(self):
        # Get Camera position (mounted on robot)
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_mat = p.getMatrixFromQuaternion(orn)

        cam_pos = [pos[0], pos[1], pos[2] + 0.5]
        forward_vec = [rot_mat[0], rot_mat[3], rot_mat[6]]
        target_pos = [cam_pos[0] + forward_vec[0],
                      cam_pos[1] + forward_vec[1],
                      cam_pos[2] + forward_vec[2]]

        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])

        w, h, rgb, depth, seg = p.getCameraImage(
            CAMERA_WIDTH, CAMERA_HEIGHT,
            view_matrix, self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # 1. Convert to numpy array
        image = np.array(rgb, dtype=np.uint8)
        # 2. Reshape to include Alpha
        image = np.reshape(image, (h, w, 4))
        # 3. Slice to RGB AND force a contiguous copy for OpenCV compatibility
        image = np.ascontiguousarray(image[:, :, :3])

        return image

    def detect_target(self, image):
        """
        Simulates ML detection.
        In a real app, replace this with: prediction = model.predict(image)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, TARGET_COLOR_LOWER, TARGET_COLOR_UPPER)

        # Find contours (simulating object bounding box)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                x, y, w, h = cv2.boundingRect(largest)
                detections.append({'label': 'TARGET_BOX', 'bbox': (x, y, w, h)})

                # Visual Debug: Draw box on image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return detections, image