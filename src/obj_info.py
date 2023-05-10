import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

from geometry_msgs.msg import Pose


class ObjectInfo:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance.init = False
        return cls.instance

    def __init__(self):
        if self.init:
            return
        self.init = True
        self.obj_dict = {}
        self.grasp_type_count = {}  # 每个类型有几个可抓取的物体
        self.roi_center = None

    def new_object(self, obj_id, obj_type):
        print("新物体id=", obj_id, " 类型=", obj_type)
        self.obj_dict[obj_id] = {"obj_type": obj_type, "grasp_plans": []}

    def add_object_mask(self, obj_id, obj_mask):
        print("添加mask到物体: ", obj_id)
        # 俯视图中物体对应像素为True
        if obj_id not in self.obj_dict:
            raise Exception
        self.obj_dict[obj_id]["obj_mask"] = obj_mask

    def add_object_pointcloud(self, obj_id, point_cloud):
        print("添加点云到物体: ", obj_id)
        if obj_id not in self.obj_dict:
            raise Exception
        self.obj_dict[obj_id]["point_cloud"] = point_cloud

    def add_object_grasp(self, obj_id, plan: Pose, quality):
        print("添加抓取到物体: ", obj_id)
        if obj_id not in self.obj_dict:
            raise Exception
        self.obj_dict[obj_id]["grasp_plans"].append((plan, quality))
        obj_type = self.obj_dict[obj_id]["obj_type"]
        print("物体类型: ", obj_type)
        if obj_type in self.grasp_type_count:
            self.grasp_type_count[obj_type] += 1
        else:
            self.grasp_type_count[obj_type] = 1

    def merge_object_pointcloud(self, point_cloud, obj_type):
        print("融合类型为", obj_type, "的物体点云")
        nn = NearestNeighbors(n_neighbors=1)
        max_num_matches = 0
        max_match_id = -1
        for obj_id, obj_info in self.obj_dict.items():
            if obj_info["obj_type"] == obj_type:
                obj_point_cloud = obj_info.get("point_cloud")
                if obj_point_cloud is None:
                    continue
                nn.fit(obj_point_cloud)
                dists, _ = nn.kneighbors(point_cloud)
                num_matches = np.sum(dists < 0.01)
                if num_matches > max_num_matches:
                    max_num_matches = num_matches
                    max_match_id = obj_id

        if max_match_id != -1:
            self.obj_dict[max_match_id]["point_cloud"] = np.concatenate(
                [self.obj_dict[max_match_id]["point_cloud"], point_cloud]
            )
            print(
                "融合到匹配的物体: ",
                max_match_id,
                "匹配点数:",
                max_num_matches,
                "融合后点数:",
                len(self.obj_dict[max_match_id]["point_cloud"]),
            )
        else:
            obj_id = 1
            while obj_id in self.obj_dict:
                obj_id += 1
            print("融合未匹配，新建物体: ", obj_id, "点数:", len(point_cloud))
            self.new_object(obj_id, obj_type)
            self.add_object_pointcloud(obj_id, point_cloud)

    def get_best_grasp_plan_and_remove(
        self, obj_type
    ) -> tuple[Pose | None, npt.NDArray | None]:
        print("获取最佳抓取并移除一个类型为", obj_type, "的物体")
        best_quality = 0
        best_plan = None
        best_obj_id = None
        best_obj_type = None
        best_obj_mask = None
        for obj_id, obj_info in self.obj_dict.items():
            if obj_info["obj_type"] != obj_type:
                continue
            grasp_plans = obj_info.get("grasp_plans")
            if grasp_plans is None:
                continue
            for plan, quality in grasp_plans:
                if quality > best_quality:
                    best_quality = quality
                    best_plan = plan
                    best_obj_id = obj_id
                    best_obj_type = obj_info["obj_type"]
                    best_obj_mask = obj_info.get("obj_mask")
        print("最佳目标ID:", best_obj_id)
        if best_obj_id is None:
            return None, None
        del self.obj_dict[best_obj_id]
        self.grasp_type_count[best_obj_type] -= 1
        return best_plan, best_obj_mask

    def find_object_by_position(self, position):
        print("根据位置查找物体", position)
        nn = NearestNeighbors(n_neighbors=1)
        max_num_matches = 0
        max_match_id = -1
        for obj_id, obj_info in self.obj_dict.items():
            obj_point_cloud = obj_info["point_cloud"]
            nn.fit(obj_point_cloud)
            dists, _ = nn.kneighbors(np.expand_dims(position, 0))
            # breakpoint()
            num_matches = np.sum(dists < 0.1)
            if num_matches > max_num_matches:
                max_num_matches = num_matches
                max_match_id = obj_id
        if max_match_id != -1:
            print("找到物体", max_match_id, "匹配点数:", max_num_matches)
            return max_match_id
        return None

    def get_point_cloud(self, obj_id):
        if obj_id not in self.obj_dict:
            return None
        return self.obj_dict[obj_id].get("point_cloud")

    def get_all_pc(self):
        pcs = []
        for _, obj_info in self.obj_dict.items():
            obj_point_cloud = obj_info["point_cloud"]
            pcs.append(obj_point_cloud)
        return pcs

    def is_empty(self):
        sum = 0
        for _, count in self.grasp_type_count.items():
            sum += count
        return sum == 0

    def clear_all(self):
        self.obj_dict = {}

    def __str__(self) -> str:
        str = ""
        for obj_id, obj_info in self.obj_dict.items():
            str += f"obj_id: {obj_id}, obj_type: {obj_info['obj_type']}, points:{len(obj_info['point_cloud'])}\n"
            str += f"grasps:"
            grasps = obj_info.get("grasp_plans")
            for grasp in grasps:
                str += f"(g,q={grasp[1]:.3f})"
            str += "\n"
        str += f"grasp_type_count: {self.grasp_type_count}\n"
        str += f"roi_center: {self.roi_center}\n"
        return str
