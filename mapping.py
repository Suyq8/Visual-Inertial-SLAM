import numpy as np
from transform import pixel2world


class Map:
    def __init__(self, n_landmark, camera):
        self.n_landmark = n_landmark
        self.seen_landmark = np.array([], dtype=np.int16)
        self.landmark = np.zeros((4, self.n_landmark))  # 4*m
        self.landmark[-1, :] = 1
        self.camera = camera

    def get_seen_unseen_landmark(self, cur_idx):
        seen = np.intersect1d(self.seen_landmark, cur_idx, True)
        unseen = np.setdiff1d(cur_idx, self.seen_landmark, True)

        self.seen_landmark = np.append(self.seen_landmark, unseen)

        return seen, unseen

    def initialize_new_landmark(self, unseen, pose):
        if len(unseen) > 0:
            unseen_feature = self.camera.get_feature(unseen)
            world_xyz = pixel2world(
                unseen_feature, pose, self.camera.imu_T_cam, self.camera.M, self.camera.b)
            self.landmark[:3, unseen] = world_xyz[:3, :]

    def update_old_landmark_mean(self, seen, delta):
        self.landmark[:3, seen] += delta.reshape((3, -1), order="F")

    def get_landmark(self, idx):
        return self.landmark[:, idx]
