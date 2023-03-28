
### Example use
### enemyList, obstaclesList, powerupsList = binaryWorldMaker(self.observation, dictionaryList)

#dictionaryList ={'enemies':[2,9,25,20],'obstacles':[-10,-11],'powerups':[16,21]}
dictionaryList ={'enemies':[2,9,25,20],'obstacles':[246,245],'powerups':[16,21]}

def binaryWorldMaker(state, dictionary):
    ## Get the world representation
    state[4] = state

    # Enemies
    enemyList = []
    for i in range(0, 21):
        enemyList.append([0] * 22)

    counter = 0
    for i in state:
        for e in dictionary.get('enemies'):
            idList = [v for v, val in enumerate(i) if val == e]
            for location in idList:
                enemyList[counter][location] = 1
        counter += 1

    # Obstacles
    obstaclesList = []
    for i in range(0, 21):
        obstaclesList.append([0] * 22)

    counter = 0
    for i in state:
        for e in dictionary.get('obstacles'):
            idList = [v for v, val in enumerate(i) if val == e]
            for location in idList:
                obstaclesList[counter][location] = 1
        counter += 1

    # Powerups
    powerupsList = []
    for i in range(0, 21):
        powerupsList.append([0] * 22)

    counter = 0
    for i in state:
        for e in dictionary.get('powerups'):
            idList = [v for v, val in enumerate(i) if val == e]
            for location in idList:
                powerupsList[counter][location] = 1
        counter += 1

    return enemyList, obstaclesList, powerupsList
