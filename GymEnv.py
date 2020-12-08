import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet
import pybullet_utils.bullet_client as bc
from pybullet_utils import gazebo_world_parser
from . import racecar
import random
import pybullet_data
from pkg_resources import parse_version

import torch
import torch.distributions.normal as N

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

randomMap = True


class RacecarZEDGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=10,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=True):
    print("init")
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._isDiscrete = isDiscrete
    self._renders = renders
    self._timeStep = 0.01   # default is 240 Hz
    self._envStepCounter = 0
    self._prev_envStepCounter = 0
    self._terminationNum = 1000 # max number of actions before reset
    self._cam_dist = 20
    self._cam_yaw = 50
    self._cam_pitch = -35
    self._width = 128     # 640 -> camera width
    self._height = 128    # 480 -> camera height
    self._prev_distance = 0
    self._prevState = "red"

    if self._renders:
      self._p = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._p = bc.BulletClient()

    # objects
    self._finishLineUniqueId = None
    self._mapObjects = None
    self._staticObjects = {}

    # reset
    self.seed()
    self.reset()
    self.addDebug()

    observationDim = len(self.getExtendedObservation())

    observation_high = np.array([np.finfo(np.float32).max] * observationDim)

    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    self.observation_space = spaces.Box(low=0,
                                        high=255,
                                        shape=(self._height, self._width, 4),
                                        dtype=np.uint8)

    self.viewer = None


  def reset(self):

    self._p.resetSimulation()
    #p.setPhysicsEngineParameter(numSolverIterations=300)
    self._p.setTimeStep(self._timeStep)
    self._p.setGravity(0, 0, -9.8)                  # 9.8 in minus Z direction

    # load the map
    self._mapObjects = gazebo_world_parser.parseWorld(self._p, filepath=os.path.join(self._urdfRoot, "OBJs/gazebo/worlds/racetrack_day.world"))

    # set finish line
    self._finishLineUniqueId = 5     # aws_robomaker_racetrack_Billboard_01 (hard coded)

    # select static objects
    for k, v in self._mapObjects.items():  # nested dict
        if k[-1] == "S":
            self._staticObjects.update({k: v})

    '''
    mapObjects = self._p.loadSDF(os.path.join(self._urdfRoot, "buggy.sdf"))
    for i in mapObjects:
      pos, orn = self._p.getBasePositionAndOrientation(i)
      newpos = [pos[0], pos[1], pos[2] + 0.1]                   # move the map objects slightly above 0
      self._p.resetBasePositionAndOrientation(i, newpos, orn)   # reset positions and orientations (center of mass)
    
    print(self._p.loadURDF(os.path.join(self._urdfRoot, "r2d2.urdf"), [finishLineX, finishLineY, finishLineZ]))
    '''

    # load the racecar
    self._racecar = racecar.Racecar(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

    if randomMap:
        newX = -10 + 4. * random.random()
        newY = -15.5
    else:
        newX = -10
        newY = -15.5

    newZ = pos[2]
    self._p.resetBasePositionAndOrientation(self._racecar.racecarUniqueId, [newX, newY, newZ], orn)
    self._p.resetDebugVisualizerCamera(1, -90, -40, [newX, newY, newZ])

    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()

    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def __del__(self):
    self._p = 0

  def addDebug(self):
    # add button
    self.button_id = self._p.addUserDebugParameter("button", 1, 0, 1)   # bug, still not working well

    # add slider
    self.slider_id = self._p.addUserDebugParameter("slider", 0.0, 1.0, 0.0)

  def updateDebug(self):
    """called in step function"""
    # read button
    randomMap = self._p.readUserDebugParameter(self.button_id)

    # read slider
    dummy = self._p.readUserDebugParameter(self.slider_id)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    carmat = self._p.getMatrixFromQuaternion(carorn)
    finishLinepos, finishLineorn = self._p.getBasePositionAndOrientation(self._finishLineUniqueId)
    invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
    finishLinePosInCar, finishLineOrnInCar = self._p.multiplyTransforms(invCarPos,
                                                                        invCarOrn,
                                                                        finishLinepos,
                                                                        finishLineorn)
    dist0 = 0.3
    dist1 = 1.
    eyePos = [
        carpos[0] + dist0 * carmat[0], carpos[1] + dist0 * carmat[3],
        carpos[2] + dist0 * carmat[6] + 0.3
    ]
    targetPos = [
        carpos[0] + dist1 * carmat[0], carpos[1] + dist1 * carmat[3],
        carpos[2] + dist1 * carmat[6] + 0.3
    ]
    up = [carmat[2], carmat[5], carmat[8]]
    viewMat = self._p.computeViewMatrix(eyePos, targetPos, up)
    #viewMat = self._p.computeViewMatrixFromYawPitchRoll(carpos,1,0,0,0,2)

    projMatrix = [
        0.7499999403953552, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, -1.0000200271606445, -1.0,
        0.0, 0.0, -0.02000020071864128, 0.0]
    img_arr = self._p.getCameraImage(width=self._width,
                                     height=self._height,
                                     viewMatrix=viewMat,
                                     projectionMatrix=projMatrix)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    self._observation = np_img_arr
    return self._observation

  def step(self, action):
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
      # self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)     # a button for this

    if (self._isDiscrete):
      fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
      steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
      forward = fwd[action]
      steer = steerings[action]
      realaction = [forward, steer]
    else:
      realaction = action

    self._racecar.applyAction(realaction)

    for i in range(self._actionRepeat):
      self._p.stepSimulation()

      if self._renders:
        time.sleep(self._timeStep)

      self._observation = self.getExtendedObservation()

      if self._termination():
        break

      self._envStepCounter += 1

    reward = self._reward()
    done = self._termination()  # resets the environment
    #print("len=%r" % len(self._observation))

    self.updateDebug()

    return np.array(self._observation), reward, done, {}

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    return self._envStepCounter > self._terminationNum

  def _distance2D(self, posA, posB):
      return math.sqrt((posA[0]-posB[0])**2 + (posA[1]-posB[1])**2)

  def _distance3D(self, posA, posB):
      return math.sqrt((posA[0]-posB[0])**2 + (posA[1]-posB[1])**2 + (posA[2]-posB[2])**2)

  def _distanceFromTrafficLine(self, car_pos, center, length, width, radius):
      UpperX = center[0] + length/2
      BottomX = center[0] - length/2
      if car_pos[0] < UpperX and car_pos[0] > BottomX:  # the car is in one of the straight lines
          distance = abs(car_pos[1]) - width
      else:                                             # the car is in one of the corners
          if car_pos[0] >= UpperX:                      # deciding which corner
              r = self._distance2D(car_pos, [UpperX, 0])
          else:
              r = self._distance2D(car_pos, [BottomX, 0])
          distance = r - radius

      return distance

  def _trafficLightStateMachine(self):
      state = self._prevState

      if ((self._envStepCounter - self._prev_envStepCounter) >= 200) and self._prevState == "green":
          state = "red"
          self._prev_envStepCounter = self._envStepCounter
          self._prevState = state

      if ((self._envStepCounter - self._prev_envStepCounter) >= 50) and self._prevState == "red":
          state = "green"
          self._prev_envStepCounter = self._envStepCounter
          self._prevState = state

      return state

  def _reward(self):
    # initial reward
    alpha = -1000   # continuous reward: distance from the finish line (linear)
    beta = 0        # continuous reward: positon in a lane (gauss)
    gamma = 0       # discrete reward: stopping at traffic lights (linear)
    delta = 0       # discrete reward: avoiding static objects (1/x)
    epsilon = 0     # discrete reward: avoiding moving objects (1/x)

    # weights for the different rewards, these should be tuned via sliders
    w0 = 1
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    w5 = 0

    """timing should be tau -> end of the episode
    if got there: positive reward and new finish line"""

    # get car positions
    car_pos, car_orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)


    # alpha
    finish_pos, finish_orn = self._p.getBasePositionAndOrientation(self._finishLineUniqueId)
    car_fin_dist = self._distance3D(car_pos, finish_pos)
    alpha = -(car_fin_dist - self._prev_distance)      # distance difference
    self._prev_distance = car_fin_dist

    # beta
    center = self._mapObjects["aws_robomaker_racetrack_Trackline_01"]["pose_xyz"]
    x = self._distanceFromTrafficLine(car_pos=car_pos,
                                      center=center,
                                      length=32,
                                      width=15,         # width should be the same as radius
                                      radius=14.75)     # traffic line is the zero point
    distBetweenLanes = 1.2          # hard coded
    mu = distBetweenLanes/2.0       # maybe shift left?
    sig = mu/3.0                    # 3 sigma -> 99.7%
    m = N.Normal(torch.tensor([mu]), torch.tensor([sig]))
    beta = m.log_prob(x).item()     # = ln(Normal)-> returns tensor      ? saturation

    # gamma
    tLight_pos = self._mapObjects["aws_robomaker_racetrack_RaceStartLight_01_00"]["pose_xyz"]
    car_light_dist = self._distance2D(car_pos, tLight_pos)
    velocity = self._p.getBaseVelocity(self._racecar.racecarUniqueId)
    tLight = self._trafficLightStateMachine()

    if tLight == "green":
      gamma = 0
    else:
        if (car_light_dist > 0.5 and car_light_dist < 2):                       # agent stops within 2 meters
            velocity_vector = math.sqrt(velocity[0][0] ** 2 + velocity[0][1] ** 2)  # sqrt(vx^2 + vy^2)
            gamma = -velocity_vector

        elif (car_light_dist < 0.5):                          # agent is too close
            gamma = -100                                        # huge error

        else:
            gamma = 0

    # delta
    nearest = 1000
    sObj_pos = []

    for i, k in enumerate(self._staticObjects.keys()):
        sObj_pos.append(self._staticObjects[k]["pose_xyz"])
        car_sObj_dist = self._distance3D(car_pos, sObj_pos[i])

        if car_sObj_dist < nearest:
            nearest = car_sObj_dist

    if nearest < 1:
        delta = - math.exp(-(nearest**2))
    else:
        delta = 0

    # epsilon
    if False:
        # moving objects are other urdfs -> this function can be used
        mObjDist = self._p.getClosestPoints(self._racecar.racecarUniqueId,
                                            movingObjUniqueId,
                                            10000)  # this is the max allowed distance

        if mObjDist < 1:
            m = N.Normal(torch.tensor([0.0]), torch.tensor([1.0]))  # saturation - > gauss: mu-objektum -> 0
            epsilon = 1 / m.log_prob(mObjDist[0][8])
        else:
            epsilon = 0

    reward = w0*(w1*alpha + w2*beta + w3*gamma + w4*delta + w5*epsilon)
    print("reward:", reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
