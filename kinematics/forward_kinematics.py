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

from numpy.matlib import matrix, identity
import numpy as np
from recognize_posture import PostureRecognitionAgent
import pandas as pd
from io import StringIO
def Traf3d(roll,pitch,yaw,dx,dy,dz):
    return np.r_[np.c_[RollPitchYawMatrix(roll,pitch,yaw),np.array([dx,dy,dz]).reshape((3,1))],np.c_[np.zeros((1,3)),1]]
df = pd.read_csv(StringIO("""HeadYaw,        2,0,0,0.1265
HeadPitch,      1,0,0,0
LShoulderPitch, 1,0,0.098,0.1
LShoulderRoll,  0,0,0,0
LElbowYaw,      2,0.105,0.015,0
LElbowRoll,     0,0,0,0
LWristYaw2,     2,0.05595,0,0
LHipYawPitch1,  1,0,0.05,-0.085
LHipRoll,       0,0,0,0
LHipPitch,      1,0,0,0
LKneePitch,     1,0,0,-0.1
LAnklePitch,    1,0,0,-0.1029
LAnkleRoll,     0,0,0,0
RHipYawPitch1,  1,0,-0.05,-0.085
RHipRoll,       0,0,0,0
RHipPitch,      1,0,0,0
RKneePitch,     1,0,0,-0.1
RAnklePitch,    1,0,0,-0.1029
RAnkleRoll,     0,0,0,0
RShoulderPitch, 1,0,-0.098,0.1
RShoulderRoll,  0,0,0,0
RElbowYaw,      2,0.105,-0.015,0
RElbowRoll,     0,0,0,0
RWristYaw2,     2,0.05595,0,0"""), header=None)
jointAngleLinksDict=dict(map(lambda a:(a[0],a[1:]),df.values))
def RollPitchYawMatrix(roll,pitch,yaw):
    return [[np.cos(roll)* np.cos(pitch),np.sin(roll) *-np.cos(pitch),np.sin(pitch)],
           [np.sin(yaw)* np.cos(roll)* np.sin(pitch) +  np.cos(yaw) * np.sin(roll),np.cos(yaw) *np.cos(roll)-np.sin(yaw) *np.sin(roll)* np.sin(pitch),np.sin(yaw)* -np.cos(pitch)],
           [np.sin(yaw) *np.sin(roll)-np.cos(yaw) *np.cos(roll) *np.sin(pitch),np.cos(yaw) *np.sin(roll) *np.sin(pitch)+np.sin(yaw) *np.cos(roll),np.cos(yaw) *np.cos(pitch)]]

class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       # YOUR CODE HERE
                       'LArm':['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw2'],
                       'LLeg':['LHipYawPitch1', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'RAnkleRoll'],
                       'RLeg':['RHipYawPitch1', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'LAnkleRoll'],
                       'RArm':['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw2']
                       }

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
        #T = identity(4)
        # YOUR CODE HERE
        T = (lambda _:Traf3d(*np.roll(np.eye(1,3), _[0]).flatten()*joint_angle,*_[1:]))(jointAngleLinksDict[joint_name])
        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                # YOUR CODE HERE
                T*=T1
                self.transforms[joint] = T

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
