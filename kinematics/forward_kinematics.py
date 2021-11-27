'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

import numpy as np
from math import sin, cos, pi

from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: np.identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
                       }
        # http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
        # Defined in mm
        links = {
            'HeadYaw':        (  0.00,  0.00,  126.50),
            'HeadPitch':      (  0.00,  0.00,    0.00),
            'LShoulderPitch': (  0.00, 98.00,  100.00),
            'LShoulderRoll':  (  0.00,  0.00,    0.00),
            'LElbowYaw':      (105.00, 15.00,    0.00),
            'LElbowRoll':     (  0.00,  0.00,    0.00),
            'LWristYaw':      ( 55.95,  0.00,    0.00),
            'LHipYawPitch':   (  0.00, 50.00,  -85.00),
            'LHipRoll':       (  0.00,  0.00,    0.00),
            'LHipPitch':      (  0.00,  0.00,    0.00),
            'LKneePitch':     (  0.00,  0.00, -100.00),
            'LAnklePitch':    (  0.00,  0.00, -102.90),
            'LAnkleRoll':     (  0.00,  0.00,    0.00),
        }
        # mm to m
        self.links = {k: tuple([i / 1000 for i in v]) for k, v in links.items()}
        for k, v in list(self.links.items()):
            if k[0] == 'L':
                self.links[f"R{k[1:]}"] = (v[0], -v[1], v[2])
        link_quats = {
            'HeadYaw':        (0, 0, 1),
            'HeadPitch':      (0, 1, 0),
            'LShoulderPitch': (0, 1, 0),
            'LShoulderRoll':  (0, 0, 1),
            'LElbowYaw':      (1, 0, 0),
            'LElbowRoll':     (0, 0, 1),
            'LWristYaw':      (1, 0, 0),
            'LHipYawPitch':   (0,-1, 1),
            'LHipRoll':       (1, 0, 0),
            'LHipPitch':      (0, 1, 0),
            'LKneePitch':     (0, 1, 0),
            'LAnklePitch':    (0, 1, 0),
            'LAnkleRoll':     (1, 0, 0),
        }
        self.link_quats = {k: tuple([x / np.linalg.norm(v) for x in v]) for k,v in link_quats.items()}
        for k, v in list(self.link_quats.items()):
            if k[0] == 'L':
                self.link_quats[f"R{k[1:]}"] = (v[0], -v[1], v[2])

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        r = np.array([[0], [0], [0], [0]], np.float32)
        for i in range(3):
            r[i, 0] = self.link_quats[joint_name][i]
        r = r / np.linalg.norm(r)

        # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        phi = joint_angle
        m = r * r.T * (1 - cos(phi))
        m += np.array([
            [           cos(phi), -r[2, 0] * sin(phi),  r[1, 0] * sin(phi), 0],
            [ r[2, 0] * sin(phi),            cos(phi), -r[0, 0] * sin(phi), 0],
            [-r[1, 0] * sin(phi),  r[0, 0] * sin(phi),            cos(phi), 0],
            [                  0,                   0,                   0, 1],
        ], np.float32)

        t = np.array([
            [1, 0, 0, self.links[joint_name][0]],
            [0, 1, 0, self.links[joint_name][1]],
            [0, 0, 1, self.links[joint_name][2]],
            [0, 0, 0,                         1],
        ], np.float32)

        return t @ m

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = np.identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                T = T @ Tl

                self.transforms[joint] = T

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
