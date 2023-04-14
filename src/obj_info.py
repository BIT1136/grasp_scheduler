import numpy as np
from sklearn.neighbors import NearestNeighbors


class ObjectInfo:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        self.obj_dict = {}
        self.grasp_type_count = {}

    def new_object(self, obj_id, obj_type):
        self.obj_dict[obj_id] = {"id": obj_id, "obj_type": obj_type, "grasp_plans": []}

    def add_object_pointcloud(self, obj_id, point_cloud):
        if obj_id not in self.obj_dict:
            raise Exception
        self.obj_dict[obj_id]["point_cloud"] = point_cloud

    def add_object_grasp(self, obj_id, plan, quality):
        if obj_id not in self.obj_dict:
            raise Exception
        if self.obj_dict[obj_id].get("grasp_plans") is not None:
            self.obj_dict[obj_id]["grasp_plans"].append((plan, quality))
        else:
            self.obj_dict[obj_id]["grasp_plans"] = []
            obj_type=self.obj_dict[obj_id]["obj_type"]
            if obj_type in self.grasp_type_count:
                self.grasp_type_count[obj_type] += 1
            else:
                self.grasp_type_count[obj_type] = 0

    def merge_object_pointcloud(self, point_cloud, obj_type):
        match_found = False
        for obj_id, obj_info in self.obj_dict.items():
            if obj_info.get("obj_type") != obj_type:
                continue
            obj_point_cloud = obj_info.get("point_cloud")
            if obj_point_cloud is None:
                continue
            distances, _ = (
                NearestNeighbors(n_neighbors=1)
                .fit(obj_point_cloud)
                .kneighbors(point_cloud)
            )
            if np.min(distances) < 10:
                self.obj_dict[obj_id]["point_cloud"] = np.concatenate(
                    [obj_point_cloud, point_cloud]
                )
                match_found = True
                break
        if not match_found:
            obj_id = 1
            while obj_id in self.obj_dict:
                obj_id += 1
            self.new_object(obj_id, obj_type)
            self.add_object_pointcloud(obj_id, point_cloud)

    def get_best_grasp_plan_and_remove(self, obj_type):
        best_quality = -1
        best_plan = None
        best_obj_id = None
        best_obj_type = None
        for obj_id, obj_info in self.obj_dict.items():
            if obj_info.get("obj_type") != obj_type:
                continue
            grasp_plans = obj_info.get("grasp_plans")
            if grasp_plans is None:
                continue
            for plan, quality in grasp_plans:
                if quality > best_quality:
                    best_quality = quality
                    best_plan = plan
                    best_obj_id = obj_id
                    best_obj_type = obj_info.get("obj_type")
        if best_obj_id is None:
            return None
        del self.obj_dict[best_obj_id]
        self.grasp_type_count[best_obj_type] -= 1
        return best_plan

    def find_object_by_position(self, position):
        for obj_id, obj_info in self.obj_dict.items():
            obj_point_cloud = obj_info.get("point_cloud")
            if obj_point_cloud is None:
                continue
            distances, _ = (
                NearestNeighbors(n_neighbors=1)
                .fit(obj_point_cloud)
                .kneighbors(position)
            )
            if np.min(distances) < 10:
                return obj_id
        return None

    def get_point_cloud(self, obj_id):
        if obj_id not in self.obj_dict:
            return None
        return self.obj_dict[obj_id]["point_cloud"]

    def is_empty(self):
        sum=0
        for _, count in self.grasp_type_count.items():
            sum += count
        return sum == 0

    def clear_all(self):
        self.obj_dict = {}
