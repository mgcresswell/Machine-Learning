import numpy as np
import random 
import RandomEnviroment as movieEnv
import pandas as pd
import numpy.ma as ma
import time




#Preform random recommendations on datasets
#for baseline comparison
for x in range(50):
    env2 = movieEnv.RandomMovieEnviroment([2858,260,1196,1210,480], 1000)
    action_size = env2.getActionSpaceSize()
    state_size = env2.getStateSpaceSize()
    qtable = env2.createQTable()

    rewards = []
    total_episodes = 0
    totalSteps = 0
    steps = 0
    stepMetric = 0
    env2.resetTraining()
    tic = time.perf_counter()  
    while (env2.keepTraining):        
        userState = env2.getTrainingUserWithState()   
        for step in range(state_size):      
            curUserStateValues, curUserStateIndexes = env2.setUserState(userState, step)
            newState = False
            while (newState == False):            
            
                action = env2.actionspace_sample(curUserStateValues)
                new_curUserStateValues, new_curUserStateIndexes, reward, maxReward, metric, newState = env2.testStep(curUserStateValues, curUserStateIndexes, action, userState)             

                curUserStateValues = new_curUserStateValues
                curUserStateIndexes = new_curUserStateIndexes
                totalSteps += 1
                stepMetric += metric

    print(f"{(stepMetric/totalSteps)*100:0.2f}%")

        









