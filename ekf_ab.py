import numpy as np
from scipy.linalg import expm
from mapping import Map
from transform import q_derivative, circle, world2pixel, hat


class ExtendedKalmanFilter_ab:
    def __init__(self, time_length, n_landmark, camera, prior_cov_pose=0.01, prior_cov_landmark=0.1):
        self.mu = np.zeros((4, 4, time_length))  # pose
        self.mu[:, :, 0] = np.eye(4)
        self.sigma_pose = np.eye(6)*prior_cov_pose
        self.sigma_landmark = np.eye(3*n_landmark)*prior_cov_landmark
        self.landmark = Map(n_landmark, camera)
        self.idx = 0
        self.camera = camera
        self.P = np.eye(3, 4)

    def predict(self, tau, u_hat, u_bhat, w):
        self.mu[:, :, self.idx+1] = self.mu[:, :, self.idx] @ expm(tau*u_hat)
        F = expm(-tau*u_bhat)
        self.sigma_pose = F @ self.sigma_pose[:6, :6] @ F.T + w

    def update(self):
        cur_idx = self.camera.get_cur_landmark()
        seen, unseen = self.landmark.get_seen_unseen_landmark(cur_idx)

        self.landmark.initialize_new_landmark(
            unseen, self.mu[:, :, self.idx+1])

        if len(seen) > 0:
            # update seen landmark
            H, innovation = self.cal_H_innovation(seen)  # 4n*(6+3m) 4n*1

            idx_seen = np.array([i for n in seen for i in (3*n, 3*n+1, 3*n+2)])
            sigma_seen = self.sigma_landmark[np.ix_(idx_seen, idx_seen)]
            K = self.cal_K(H, len(seen), sigma_seen)

            delta_mu = K@innovation  # (6+3n)*1

            # update landmark
            self.landmark.update_old_landmark_mean(seen, delta_mu)

            # update covariance
            self.sigma_landmark[np.ix_(idx_seen, idx_seen)] = (
                np.eye(sigma_seen.shape[0])-K@H)@sigma_seen

    def update_idx(self):
        self.idx += 1

    def cal_H_innovation(self, idx):
        # pose: 4*4@n*4*4@4*4@n*4*6->n*4*6->4n*6
        # landmark: n*4*4@4*4@4*3->n*4*3->4n*3m(3n)
        pose_inv = np.linalg.inv(self.mu[:, :, self.idx+1])

        landmark_world = self.landmark.get_landmark(idx)  # 4*n
        landmark_imu = pose_inv @ landmark_world  # 4*n
        q = self.camera.cam_T_imu @ landmark_imu  # 4*n
        q_prime = q_derivative(q)  # n*4*4

        tmp = np.einsum('ij,kjl->kil', self.camera.M, q_prime)  # n*4*4
        tmp = np.einsum('ijk,kl->ijl', tmp, self.camera.cam_T_imu)  # n*4*4

        H_landmark = np.zeros(
            (4*landmark_world.shape[1], 3*landmark_world.shape[1]))  # 4n*3m(3n)
        H_part = np.einsum('ijk,kl->ijl', tmp, pose_inv @ self.P.T)

        for i in range(len(idx)):
            H_landmark[i*4:(i+1)*4, i * 3:(i+1)*3] = H_part[i, :, :]

        observation = self.camera.get_feature(idx)
        innovation = observation - \
            world2pixel(landmark_world, pose_inv,
                        self.camera.cam_T_imu, self.camera.M)  # 4*n->4n*1

        return H_landmark, innovation.ravel('F').reshape(-1, 1)

    def cal_K(self, H, cur_size, sigma_seen):
        # (6+3m)*(6+3m)@(6+3m)*4n@((4n)*(6+3m)@(6+3m)*(6+3m)@(6+3m)*4n+4n*4n)^-1->(6+3m)*4n
        # (6+3n)*(6+3n)@(6+3n)*4n@((4n)*(6+3n)@(6+3n)*(6+3n)@(6+3n)*4n+4n*4n)^-1->(6+3n)*4n
        return sigma_seen@H.T@np.linalg.pinv(H@sigma_seen@H.T + np.kron(np.eye(cur_size), self.camera.v))
