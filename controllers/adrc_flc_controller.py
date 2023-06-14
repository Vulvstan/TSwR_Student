import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([[3 * p[0], 0], [0, 3 * p[1]], [3 * p[0] ** 2, 0],
                           [0, 3 * p[1] ** 2], [p[0] ** 3, 0], [0, p[1] ** 3]])
        W = np.hstack((np.eye(2), np.zeros((2, 4))))
        A = np.eye(6, k=2)
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        inv_M = np.linalg.inv(M)
        C = self.model.C(x)
        A = np.eye(6, k=2)
        B = np.zeros((6, 2))

        A[2:4, 2:4] = -inv_M @ C
        B[2:4, :] = inv_M


        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        return NotImplementedError
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        M = self.model.M(x)
        C = self.model.C(x)

        z_estimate = self.eso.get_state()
        x_estimate = z_estimate[0:2]
        x_estimate_dot = z_estimate[2:4]
        f = z_estimate[4:]

        v = q_d_ddot + self.Kd @ (q_d_dot - x_est_dot) + self.Kp @ (q_d - q)
        u = self.model.M(x) @ (v - f) + self.model.C(x) @ x_est_dot

        self.update_params(x_estimate, x_estimate_dot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))

        return u