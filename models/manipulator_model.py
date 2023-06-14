import numpy as np


class ManipulatorModel:
    def __init__(self, Tp, m3=0.0, r3=0.01):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        # odległości do środka masy
        d1 = self.l1 / 2
        d2 = self.l2 / 2

        alhpa = self.I_1 + self.I_2 + self.I_3 + self.m1 * d1 ** 2 + self.m2 * (
                self.l1 ** 2 + d2 ** 2) + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = self.I_2 + self.m2 * (self.l2 / 2) ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        m_1_1 = alhpa + 2 * beta * np.cos(q2)
        m_1_2 = gamma + beta * np.cos(q2)
        m_2_1 = gamma + beta * np.cos(q2)
        m_2_2 = gamma
        M = np.array([[m_1_1, m_1_2], [m_2_1, m_2_2]])
        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        # odległości do środka masy 2
        d2 = self.l2 / 2

        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        c_1_1 = -beta * np.sin(q2) * q2_dot
        c_1_2 = -beta * np.sin(q2) * (q1_dot + q2_dot)
        c_2_1 = beta * np.sin(q2) * q1_dot
        c_2_2 = 0
        C = np.array([[c_1_1, c_1_2], [c_2_1, c_2_2]])
        return C

    def x_dot(self, x, u):
        inv_M = np.linalg.inv(self.M(x))
        empty_M = np.zeros((2, 2), dtype=np.float32)

        A = np.concatenate([np.concatenate([empty_M, np.eye(2)], 1), np.concatenate([empty_M, -inv_M @ self.C(x)], 1)],
                           0)
        b = np.concatenate([empty_M, inv_M], 0)

        x_dot = A @ x[:, np.newaxis] + b @ u
        return x_dot
