__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 13, 2009 1:29:41 AM$"

import random
from client.tcpenvironment import TCPEnvironment
from utils.dataadaptor import extractObservation

class MarioEnvironment(TCPEnvironment):
    """ An Environment class, wrapping access to the MarioServer, 
    and allowing interactions to a level. """

    # Level settings
    levelDifficulty = 2
    levelType = 0
    creaturesEnabled = True
    initMarioMode = 2
    levelSeed = 1
    randomLevelSeed = False
    timeLimit = 100
    fastTCP = False
    maxFPS = True
    
    # Other settings
    visualization = True
    otherServerArgs = ""
    numberOfFitnessValues = 5

    def getSensors(self):
        data = TCPEnvironment.getSensors(self)
#        print "data: ", data
        return extractObservation(data)

    def reset(self):

        if self.randomLevelSeed:
            self.levelSeed = random.randint(-999999, 999999)

        argstring = "-ld %d -lt %d -mm %d -ls %d -tl %d " % (self.levelDifficulty,
                                                            self.levelType,
                                                            self.initMarioMode,
                                                            self.levelSeed,
                                                            self.timeLimit
                                                            )
        if self.creaturesEnabled:
            argstring += "-pw off "
        else:
            argstring += "-pw on "

        if self.visualization:
            argstring += "-vis on "
        else:
            argstring += "-vis off "

        if self.fastTCP:
            argstring += "-fastTCP on"

        if self.maxFPS:
            argstring += "-maxFPS on"
        else:
            argstring += "-maxFPS off" 

        self.client.sendData("reset " + argstring + self.otherServerArgs + "\r\n")

