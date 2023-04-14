from dataclasses import dataclass

import numpy as np

from sensor_msgs.msg import CameraInfo

@dataclass
class CameraIntrinsic:
    fx:float
    fy:float
    cx:float
    cy:float

    @classmethod
    def from_camera_info(cls,camera_info: CameraInfo):
        return cls(camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5])
    
    def as_matrix(self):
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]