'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[[float, [int, float, float], [int, float, float]], ...],[[float, [int, float, float], [int, float, float]], ...],...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'software_installation'))

from pid import PIDAgent
from keyframes import hello

def binary_search(left, right, val_fun, value):
    assert left <= right

    if value < val_fun(left):
        return left

    if isinstance(left, int) and isinstance(right, int):
        def mid_fun(l, r): return (l + r) // 2 + (l + r) % 2
        cmp_eps = 0
        edge_eps = 1
    else:
        def mid_fun(l, r): return (l + r) / 2
        cmp_eps = 0.000001
        edge_eps = 0.0

    while right - left > cmp_eps:
        mid = mid_fun(left, right)
        mid_val = val_fun(mid)
        if value < mid_val:
            right = mid - edge_eps
        else:
            left = mid
    assert val_fun(left) <= value
    return left


class Point():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self):
        return Point(-self.x, -self.y)

    def __mul__(self, f):
        return Point(f * self.x, f * self.y)

    def __rmul__(self, f):
        return self.__mul__(f)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


def calc_bezier(t, points):
    assert 0 <= t <= 1
    # Slide #15
    return (1 - t) ** 3 * points[0] \
        + 3 * (1 - t) ** 2 * t * points[1] \
        + 3 * (1 - t) * t ** 2 * points[2] \
        + t ** 3 * points[3]


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.animation_timer=0
        self.time=0

    def think(self, perception):
        self.time=perception.time
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)
    
    def start_animation(self,t):
        self.animation_timer=t

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        # YOUR CODE HERE
        timer=perception.time-self.animation_timer
        #binary_search(0,len(keyframes[1][0]),)
        z=list(zip(*keyframes))
        #print(z)
        for joint in z:
            #aprint(perception.time)
            #get target of n
            name,time,keys=joint
            t=binary_search(0, len(time)-1, lambda a:time[a], timer)
            if t==0 or t==len(time)-1:
                continue
            
            P0 = Point(time[t], keys[t][0])
            P3 = Point(time[t + 1], keys[t + 1][0])
            P1 = P0 + Point(keys[t][2][1], keys[t][2][2])
            P2 = P3 + Point(keys[t + 1][1][1], keys[t + 1][1][2])
            bezier_points = [P0, P1, P2, P3]
            target_joints[name]=calc_bezier(binary_search(0.0, 1.0, 
            lambda a:calc_bezier(a,bezier_points).x, timer),bezier_points).y
            
            

        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.start_animation(365)
    agent.run()
