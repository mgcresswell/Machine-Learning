import numpy as np
import random 
import Enviroment as movieEnv
import pandas as pd
import numpy.ma as ma
import time

def ExecuteMovieRecommender(states, epsilon, max_epsilon, min_epsilon, decay_rate, totalUsers):

    enviroment = movieEnv.MovieEnviroment(states, totalUsers)
    action_size = enviroment.getActionSpaceSize()
    state_size = enviroment.getStateSpaceSize()
    qtable = enviroment.createQTable()

    learning_rate = 0.2          
    gamma = 0.95                 

    epsilon = epsilon
    max_epsilon = max_epsilon
    min_epsilon = min_epsilon
    decay_rate = decay_rate

    rewards = []
    total_episodes = 0
    totalSteps = 0
    steps = 0
    tic = time.perf_counter() 
    trainingUserCount = enviroment.availableTrainingUsers
    #Train Q-Table
    #Users are similar to episodes
    for user in range(trainingUserCount):      
        userState = enviroment.getTrainingUserWithState(user)
        total_rewards = 0
        #simulate a user being in each state
        for step in range(state_size):      
            curUserStateIndexes = enviroment.setUserState(userState, step)
            curStateIndex = enviroment.getStateIndex(curUserStateIndexes[0])
            newState = False
            #while user has unseen movies in enviroment
            while (newState == False):
                exp_exp_tradeoff = random.uniform(0, 1)
                total_episodes =+ 1
                actionIndex = None

                if exp_exp_tradeoff > epsilon:       
                    row = qtable[curStateIndex,:]
                    actionIndex = enviroment.getQTableAction(curUserStateIndexes, row)
                else:
                    actionIndex = enviroment.actionspace_sample(curUserStateIndexes)
            
                new_curUserStateIndexes, reward, maxReward, metric, newState = enviroment.step(curUserStateIndexes, actionIndex, userState)            
                nextStateIndex = enviroment.getStateIndex(new_curUserStateIndexes[0])

                #Update Q-Table
                if nextStateIndex != None:
                    qtable[curStateIndex,  actionIndex] = round(qtable[curStateIndex, actionIndex] + learning_rate * (reward + gamma * np.max(np.ma.masked_invalid(qtable[nextStateIndex, :])) - qtable[curStateIndex, actionIndex]),10)  
     
                total_rewards += reward
                curUserStateIndexes = new_curUserStateIndexes
                steps = steps + 1
                
        
        totalSteps = totalSteps + 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*totalSteps) 
        rewards.append(total_rewards)
    
        


    total_maxrewards = 0
    total_rewards = 0
    metricsuccess = 0
    totalmetric = 0
    testUserCount = enviroment.availableTestUsers
    #Test Q-Table
    for user in range(testUserCount):
        userState = enviroment.getTestUserWithState(user)
        step = 0
        for step in range(state_size): 
            curUserStateIndexes = enviroment.setUserState(userState, step)
            cur_state_index = enviroment.getStateIndex(curUserStateIndexes[0])
            newState = False
            while (newState == False):        
                total_episodes =+ 1
                row = qtable[cur_state_index,:]
                actionIndex = enviroment.getQTableAction(curUserStateIndexes, row)
        
                new_curUserStateIndexes, reward, maxReward, metric, newState = enviroment.step(curUserStateIndexes, actionIndex, userState)            
                next_state_index = enviroment.getStateIndex(new_curUserStateIndexes[0])

                total_rewards += reward
                curUserStateIndexes = new_curUserStateIndexes
                steps = steps + 1
                totalmetric += 1
                metricsuccess += metric
                total_rewards += reward
                total_maxrewards += maxReward

    toc = time.perf_counter()
    totalTime = round(toc - tic, 2)
    successPercent = round(metricsuccess/totalmetric, 5)
    return totalTime, successPercent, enviroment.availableTrainingUsers, enviroment.availableTestUsers


