import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.prev_u = 0
        A = np.eye(3, k=1)
        B = np.array([[0], [self.b], [0]])
        L = np.array([[3 * p], [3 * p ** 2], [p ** 3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q = x[0]
        q_dot = x[1]
        self.eso.update(q, self.prev_u)
        q_hat, q_hat_dot, f = self.eso.get_state()

        v = self.kp * (q_d - q) + self.kd * (q_d_dot - q_hat_dot) + q_d_ddot
        u = (v - f) / self.b
        self.prev_u = u

        return u
