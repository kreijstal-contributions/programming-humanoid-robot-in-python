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
from scipy.optimize import fmin


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        end_name = self.chains[effector_name][-1]
        def err_fun(angles, target, effector_name):
            d = {j: 0.0 for j in self.joint_names}
            for i, j in enumerate(self.chains[effector_name]):
                d[j] = angles[i]
            self.forward_kinematics(d)
            return np.linalg.norm(self.transforms[end_name] - target)

        m = fmin(
            err_fun,
            np.random.rand(len(self.chains[effector_name])) / 1000,
            (transform, effector_name),
            xtol=1/10**9, ftol=1/10**9,
            disp=True
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
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
