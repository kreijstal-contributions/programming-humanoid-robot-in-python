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
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello
from spark_agent import JOINT_CMD_NAMES


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
        self.animation_start_time = -1.0
        self.animation_end_time = -1.0

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def perception_as_keyframe(self, perception,
                               dt=0.1, names=list(JOINT_CMD_NAMES.keys())):
        times = [0.0] * len(names)
        keys = [
            [perception.joint.get(j, 0), [3, dt, 0.0], [3, dt, 0.0]]
            for j in names
        ]
        return names, times, keys

    def reset_animation_time(self, start_time, speed=1.0):
        animation_start = min([self.keyframes[1][i][0]
                              for i in range(len(self.keyframes[1]))])
        animation_end = min([self.keyframes[1][i][-1]
                            for i in range(len(self.keyframes[1]))])
        for i in range(len(self.keyframes[1])):
            for j in range(len(self.keyframes[1][i])):
                self.keyframes[1][i][j] = (
                    self.keyframes[1][i][j] - animation_start) / speed + start_time
        self.animation_start_time = start_time
        self.animation_end_time = start_time + \
            (animation_end - animation_start) / speed

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        t = perception.time
        if not (self.animation_start_time <= t <= self.animation_end_time):
            return target_joints

        names, times, keys = keyframes
        for i, name in enumerate(names):
            n_times, n_keys = times[i], keys[i]
            j = binary_search(0, len(n_times) - 1, lambda i: n_times[i], t)
            if j == 0 and t <= n_times[j] or j >= len(n_times) - 1:
                target_joints[name] = n_keys[j][0]
                continue

            assert n_keys[j][2][0] == n_keys[j + 1][1][0] == 3

            P0 = Point(n_times[j + 0], n_keys[j + 0][0])
            P3 = Point(n_times[j + 1], n_keys[j + 1][0])
            P1 = P0 + Point(n_keys[j + 0][2][1], n_keys[j + 0][2][2])
            P2 = P3 + Point(n_keys[j + 1][1][1], n_keys[j + 1][1][2])
            bezier_points = [P0, P1, P2, P3]
            bezier_t = binary_search(0.0, 1.0,
                                     lambda g: calc_bezier(g, bezier_points).x,
                                     t)
            target_joints[name] = calc_bezier(bezier_t, bezier_points).y

        # In the Readme.md:
        # "The provided keyframes doesn't have joint `RHipYawPitch`, please set
        #  `RHipYawPitch` as `LHipYawPitch` which reflects the real robot."
        # This has another error: The animation "hello" doesn't even have
        # either, so try setting it and catch the exception that may result.
        try:
            target_joints["RHipYawPitch"] = target_joints["LHipYawPitch"]
        except KeyError:
            pass

        return target_joints


if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()
    agent.reset_animation_time(1.0)
    agent.run()
