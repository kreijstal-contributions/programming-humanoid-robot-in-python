'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
import numpy as np
import math
from scipy.optimize import fmin

def decompose44(mat):
    return np.array(mat[:3, 3]), np.array(mat[:3, :3])

def qconjugate(q):
    return np.array(q) * np.array([1.0, -1.0, -1.0, -1.0])

def qangle(q):
    return 2 * math.acos(max(-1, min(q[0], 1)))

# Taken from the transforms3d PyPI package (transforms3d.quaternions).
def qmult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

# Taken from the transforms3d PyPI package (transforms3d.quaternions).
def mat2quat(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        end_name = self.chains[effector_name][-1]
        position_weight = 100
        rotation_weight = 1

        def err_fun(angles, target, effector_name):
            d = {j: 0.0 for j in self.joint_names}
            for i, j in enumerate(self.chains[effector_name]):
                d[j] = angles[i]
            self.forward_kinematics(d)

            T0, R0 = decompose44(self.transforms[end_name])
            R0 = mat2quat(R0)
            T1, R1 = decompose44(target)
            R1 = mat2quat(R1)
            tdelta = T1 - T0
            rdelta = qmult(R1, qconjugate(R0))
            tdiff = np.linalg.norm(tdelta)
            rdiff = qangle(rdelta)
            return tdiff * position_weight + rdiff * rotation_weight

        m = fmin(
            err_fun,
            np.random.rand(len(self.chains[effector_name])),
            (transform, effector_name),
            xtol=10**-9, ftol=10**-9,
        )
        return m

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        angles = self.inverse_kinematics(effector_name, transform)
        names = []
        times = []
        keys = []
        for j in self.joint_names:
            names.append(j)
            times.append([0.0, 1.0])
            keys.append([0.0, [3, 0.1, 0.0], [3, 0.1, 0.0]])
            if j in self.chains[effector_name]:
                keys.append([angles[self.chains[effector_name].index(j)], [3, 0.1, 0.0], [3, 0.1, 0.0]])
            else:
                keys.append([0.0, [3, 0.1, 0.0], [3, 0.1, 0.0]])
        self.keyframes = (names, times, keys)
        self.reset_animation_time(0.0)

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = np.identity(4)
    T[1, 3] = 0.05
    T[2, 3] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
