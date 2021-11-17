'''In this exercise you need to put all code together to make the robot be able to stand up by its own.

* Task:
    complete the `StandingUpAgent.standing_up` function, e.g. call keyframe motion corresponds to current posture

'''


from os import name
from recognize_posture import PostureRecognitionAgent
from keyframes import leftBackToStand, leftBellyToStand, rightBackToStand, rightBellyToStand
import math
import random

class StandingUpAgent(PostureRecognitionAgent):
    def think(self, perception):
        self.standing_up(perception)
        return super(StandingUpAgent, self).think(perception)

    def standing_up(self, perception):
        posture = self.posture
        t = perception.time
        if t < self.animation_end_time:
            return
        if perception.time - self.stiffness_on_off_time < self.stiffness_off_cycle:
            return
        if perception.time - self.stiffness_on_off_time - self.stiffness_on_cycle + self.stiffness_off_cycle > 0.5:
            return
        if posture == 'Back':
            keyframe_fun = random.choice([leftBackToStand, rightBackToStand])
        elif posture == 'Belly':
            keyframe_fun = random.choice([leftBellyToStand, rightBellyToStand])
        else:
            return
        print(f"Preparing animation '{keyframe_fun.__name__}'")
        names, times, keys = keyframe_fun()
        _, ctimes, ckeys = self.perception_as_keyframe(perception, 0.5, names)
        for i in range(len(names)):
            times[i].insert(0, ctimes[i])
            keys[i].insert(0, ckeys[i])
        self.keyframes = (names, times, keys)
        self.reset_animation_time(t)

class TestStandingUpAgent(StandingUpAgent):
    '''this agent turns off all motor to falls down in fixed cycles
    '''
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(TestStandingUpAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.stiffness_on_off_time = 0
        self.stiffness_on_cycle = 10  # in seconds
        self.stiffness_off_cycle = 3  # in seconds

    def think(self, perception):
        action = super(TestStandingUpAgent, self).think(perception)
        time_now = perception.time
        if time_now - self.stiffness_on_off_time < self.stiffness_off_cycle:
            action.stiffness = {j: 0 for j in self.joint_names}  # turn off joints
        else:
            action.stiffness = {j: 1 for j in self.joint_names}  # turn on joints
        if time_now - self.stiffness_on_off_time > self.stiffness_on_cycle + self.stiffness_off_cycle:
            self.stiffness_on_off_time = time_now

        return action


if __name__ == '__main__':
    agent = TestStandingUpAgent()
    agent.run()
