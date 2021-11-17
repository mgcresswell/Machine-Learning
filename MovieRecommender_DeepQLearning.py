import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import gym
import time
import Enviroment as movieEnv
import MovieRecommender_DeepQLearning_Agent as deepMovie
import numpy.ma as ma
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam



def ExecuteMovieRecommender(states, epsilon, max_epsilon, min_epsilon, decay_rate, totalUsers):


    #Train Deep Q Network
    enviroment = movieEnv.MovieEnviroment(states, totalUsers)
    optimizer = Adam(learning_rate=0.01)
    agent = deepMovie.Agent(enviroment, optimizer, epsilon, max_epsilon, min_epsilon, decay_rate)
    batch_size = 32
    totalMetric = 0
    episode = 0
    tic = time.perf_counter() 
    trainingUserCount = enviroment.availableTrainingUsers
    #Users similar to episodes
    for user in range(trainingUserCount):   
        userData = enviroment.getTrainingUserWithState(user)
        total_rewards = 0
        stateSteps = 0
        stateMetric = 0   
        #Simulate a user being in each state
        for step in range(enviroment.getStateSpaceSize()):      
            stateIndexes = enviroment.setUserState(userData, step)       
            needNewState = False
            #While the user has movies they havent seen in enviroment
            while (needNewState == False):
                total_episodes =+ 1
                stateIndex = enviroment.getStateIndex(stateIndexes[0])
                stateIndexReshape = np.reshape(stateIndex, [1, 1])
                actionIndex = agent.act(enviroment, stateIndexes, stateIndexReshape)

                nextStateIndexes, reward, maxRating, metric, needNewState = enviroment.step(stateIndexes, actionIndex, userData)                 
                nextStateIndex = enviroment.getStateIndex(nextStateIndexes[0])
                nextStateIndexReshape = np.reshape(nextStateIndex, [1, 1])
                agent.store(stateIndexReshape, actionIndex, reward, nextStateIndexReshape, needNewState)

                total_rewards += reward
                totalMetric += metric
                stateMetric += metric
                stateIndexes = nextStateIndexes
                stateSteps += 1
                agent.decay(episode)

        #set weights from target network
        agent.alighn_target_model()

        #use experience replay
        if len(agent.expirience_replay) > batch_size:           
            agent.retrain(batch_size)
      
        toc = time.perf_counter()
        episode += 1


    #Test Trained Model
    total_maxrewards = 0
    total_rewards = 0
    metricsuccess = 0
    totalmetric = 0
    testUserCount = enviroment.availableTestUsers
    for user in range(testUserCount):
        userState = enviroment.getTestUserWithState(user)
        step = 0
        for step in range(enviroment.getStateSpaceSize()): 
            stateIndexes = enviroment.setUserState(userState, step)
            needNewState = False
            while (needNewState == False):                   
                stateIndex = enviroment.getStateIndex(stateIndexes[0])
                stateIndexReshape = np.reshape(stateIndex, [1, 1])
                actionIndex = agent.predict(stateIndexes, stateIndexReshape)

                nextStateIndexes, reward, maxRating, metric, needNewState = enviroment.step(stateIndexes, actionIndex, userData)             
           
                total_rewards += reward
                stateIndexes = nextStateIndexes
                step = step + 1
                totalmetric += 1
                metricsuccess += metric
                total_rewards += reward
                total_maxrewards += reward

    toc = time.perf_counter()
    totalTime = round(toc - tic, 2)
    successPercent = round(metricsuccess/totalmetric, 5)
    return totalTime, successPercent, enviroment.availableTrainingUsers, enviroment.availableTestUsers
    


