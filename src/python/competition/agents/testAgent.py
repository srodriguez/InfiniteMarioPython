print "I am a test agent"

from marioagent import MarioAgent

import numpy
import PIL.ImageGrab
import datetime
import time, os, fnmatch, shutil
import random


class TestAgent(MarioAgent):

    action = None
    actionStr = None
    KEY_JUMP = 3
    KEY_SPEED = 4
    levelScene = None
    mayMarioJump = None
    isMarioOnGround = None
    marioFloats = None
    enemiesFloats = None
    isEpisodeOver = False
    observation = None
    observationList = []
    timer = 0

    trueJumpCounter = 0;
    trueSpeedCounter = 0;

    def reset(self):
        self.isEpisodeOver = False
        self.trueJumpCounter = 0;
        self.trueSpeedCounter = 0;

    def __init__(self):
        """Constructor"""
        self.trueJumpCounter = 0
        self.trueSpeedCounter = 0
        self.action = numpy.zeros(5, int)
        self.action[1] = 1
        self.actionStr = ""

    def printObs(self):
        """for debug"""
        #print repr(self.observation)

    def exportObservsations(self):

        ssCounter = 0

        if len(self.observationList) == 0:
            self.observationList.append(self.observation)
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
            with open("Output.txt", "a") as text_file:
                text_file.write("\n")
                text_file.write("First state input " + timestamp)
                text_file.write(str(self.observation))
                text_file.write("\n")
            saveName = ("first image-" + timestamp + ".jpg")
            im = PIL.ImageGrab.grab()
            im.save(saveName)
        else:
            pass

        if self.timer == 0:
            self.timer = time.time()
        else:
            now = time.time()
            difference = int(now - self.timer)
            if difference > 1:
                self.timer = 0
                if len(self.observationList) < 40:
                    print("Current len of observation list is {}".format(len(self.observationList)))

                    self.observationList.append(self.observation)
                    t = time.localtime()
                    timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
                    with open("Output.txt", "a") as text_file:
                        text_file.write("\n")
                        text_file.write("New state input "+timestamp)
                        text_file.write(str(self.observation))
                        text_file.write("\n")
                    saveName = ("image-" + timestamp + ".jpg")
                    im = PIL.ImageGrab.grab()
                    im.save(saveName)

                else:
                    print("Observation recording is complete")
            else:
                pass



        #print(self.observationList)


    def _dangerOfGap(self):
        for x in range(9, 13):
            f = True
            for y in range(12, 22):
                if  (self.levelScene[y, x] != 0):
                    f = False
            if (f and self.levelScene[12, 11] != 0):
                return True
        return False

    def printLevelScene(self):
        ret = ""
        for x in range(22):
            tmpData = ""
            for y in range(22):
                tmpData += self.mapElToStr(self.levelScene[x][y])
            ret += "\n%s" % tmpData
        print(ret)

    def mapElToStr(self, el):
        """maps element of levelScene to str representation"""
        s = ""
        if  (el == 0):
            s = "##"
        s += "#MM#" if (el == 95) else str(el)
        while (len(s) < 4):
            s += "#"
        return s + " "

    def integrateObservation(self, obs):
        """This method stores the observation inside the agent"""
        self.observation = obs
        if (len(obs) != 6):
            self.isEpisodeOver = True
        else:
            self.mayMarioJump, self.isMarioOnGround, self.marioFloats, self.enemiesFloats, self.levelScene, dummy = obs
#        self.printLevelScene()
        self.printObs()
        self.exportObservsations()


    def getAction(self):
        """ Possible analysis of current observation and sending an action back
        """
#        print "M: mayJump: %s, onGround: %s, level[11,12]: %d, level[11,13]: %d, jc: %d" \
#            % (self.mayMarioJump, self.isMarioOnGround, self.levelScene[11,12], \
#            self.levelScene[11,13], self.trueJumpCounter)
#        if (self.isEpisodeOver):
#            return numpy.ones(5, int)

        danger = self._dangerOfGap()
        if (self.levelScene[11, 12] != 0 or \
            self.levelScene[11, 13] != 0 or danger):
            if (self.mayMarioJump or \
                (not self.isMarioOnGround and self.action[self.KEY_JUMP] == 1)):
                self.action[self.KEY_JUMP] = 1
            self.trueJumpCounter += 1
        else:
            self.action[self.KEY_JUMP] = 0;
            self.trueJumpCounter = 0

        if (self.trueJumpCounter > 16):
            self.trueJumpCounter = 0
            self.action[self.KEY_JUMP] = 0;

        self.action[self.KEY_SPEED] = danger
        return self.action