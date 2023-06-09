import numpy as np
from transform import hat


class IMU:
    def __init__(self, t, linear_velocity, angular_velocity, noise_v=0.01, noise_omega=0.01):
        self.timestamp = t
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.tau = np.diff(t)
        self.idx = 0
        self.omega_hat = hat(self.angular_velocity)  # t*3*3
        self.v_hat = hat(self.linear_velocity)  # t*3*3
        self.u_hat, self.u_bhat = self.get_u_matrices()  # t*4*4, t*6*6
        self.w = np.block([[np.eye(3)*noise_v, np.zeros((3, 3))],
                          [np.zeros((3, 3)), np.eye(3)*noise_omega]])  # process noise 6*6

    def get_data(self):
        return self.tau[0, self.idx], self.u_hat[self.idx, :, :], self.u_bhat[self.idx, :, :]

    def get_u_matrices(self):
        u_hat = np.block([[self.omega_hat, self.linear_velocity.T[:, :, None]], [
                         np.zeros((self.length, 1, 4))]])

        u_bhat = np.block([[self.omega_hat, self.v_hat],
                           [np.zeros((self.length, 3, 3)), self.omega_hat]])
        return u_hat, u_bhat

    @property
    def length(self):
        return self.timestamp.shape[1]

    def update_idx(self):
        self.idx += 1


class StereoCamera:
    def __init__(self, K, b, features, imu_T_cam, noise=5):
        self.features = features
        self.M = np.zeros((4, 4))
        self.M[:2, :3] = K[:2, :]
        self.M[2:, :3] = K[:2, :]
        self.M[2, 3] = -K[0, 0]*b  # 4*4
        self.b = b
        self.v = np.eye(4)*noise  # observation noise 4*4
        self.idx = 0
        self.features_idx = np.arange(self.length)
        self.imu_T_cam = imu_T_cam  # 4*4
        self.cam_T_imu = np.linalg.inv(imu_T_cam)

    def get_cur_landmark(self):
        valid = (self.features[0, :, self.idx] != -1)
        return self.features_idx[valid]

    def get_feature(self, feature_idx):
        return self.features[:, feature_idx, self.idx]

    def update_idx(self):
        self.idx += 1

    @property
    def length(self):
        return self.features.shape[1]
