import numpy as np


def world2pixel(world_xyz1, pose_inv, cam_T_imu, M):
    tmp = cam_T_imu @ pose_inv @ world_xyz1
    tmp /= tmp[2, :]

    return M @ tmp


def pixel2world(pixel, pose, imu_T_cam, M, b):
    tmp = np.ones((4, pixel.shape[1]))
    tmp[2, :] = M[0, 0]*b/(pixel[0, :] - pixel[2, :])
    tmp[0, :] = (pixel[0, :] - M[0, 2])/M[0, 0]*tmp[2, :]
    tmp[1, :] = (pixel[1, :] - M[1, 2])/M[1, 1]*tmp[2, :]

    return pose @ imu_T_cam @ tmp


def q_derivative(q):
    # q-4*n->n*4*4
    x = np.zeros((q.shape[1], 4, 4))
    x[:, [0, 1, 3], [0, 1, 3]] = 1
    x[:, 0, 2] = -q[0, :]/q[2, :]
    x[:, 1, 2] = -q[1, :]/q[2, :]
    x[:, 3, 2] = -q[3, :]/q[2, :]
    return x/q[2, :, None, None]


def hat(v):
    # v-4*k->k*3*3
    x = np.zeros((v.shape[1], 3, 3))
    x[:, 0, 1] = -v[2, :]
    x[:, 0, 2] = v[1, :]
    x[:, 1, 0] = v[2, :]
    x[:, 1, 2] = -v[0, :]
    x[:, 2, 0] = -v[1, :]
    x[:, 2, 1] = v[0, :]
    return x


def circle(s):
    # s-4*n->n*4*6
    x = np.zeros((s.shape[1], 4, 6))
    x[:, :3, :3] = np.eye(3)
    x[:, :3, 3:] = -hat(s)
    return x
