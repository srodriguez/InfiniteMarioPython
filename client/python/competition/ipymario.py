__author__ = "Sergey Karakovskiy, sergey at idsia dot ch"
__date__ = "$Apr 30, 2009 1:46:32 AM$"

import datetime
import sys

from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.michaelagent import MichaelAgent
from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask

# from pybrain.... episodic import EpisodicExperiment
# TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.


def main():
    agent = ForwardAgent()
    task = MarioTask(agent.name)
    exp = EpisodicExperiment(task, agent)
    print("Task Ready")
    for _ in range(999999):
        exp.doEpisodes(1)
        print(
            str(datetime.datetime.now()) + ", episode finished with task.reward = ",
            task.reward,
        )
        print("")


#    clo = CmdLineOptions(sys.argv)
#    task = MarioTask(MarioEnvironment(clo.getHost(), clo.getPort(), clo.getAgent().name))
#    exp = EpisodicExperiment(clo.getAgent(), task)
#    exp.doEpisodes(3)

if __name__ == "__main__":
    main()
else:
    print("This is module to be run rather than imported.")
