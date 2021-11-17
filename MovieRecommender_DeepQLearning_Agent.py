import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import time
import Enviroment as movieEnv
import numpy.ma as ma
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam



class Agent:
    def __init__(self, enviroment, optimizer, epsilon, max_epsilon, min_epsilon, decay_rate):
        
        self._state_size = enviroment.getStateSpaceSize()
        self._action_size = enviroment.getActionSpaceSize()
        self._optimizer = optimizer
        self.expirience_replay = deque(maxlen=2000)
        
        self.epsilon = epsilon                 
        self.max_epsilon = max_epsilon             
        self.min_epsilon = min_epsilon           
        self.decay_rate = decay_rate      
        self.gamma = 0.6              
        
        self.q_network = self.build()
        self.target_network = self.build()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)     
        for state, action, reward, next_state, terminated in minibatch:        
            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)
    
    #Deep Q Architecture
    def build(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, enviroment, actionIndexes, stateIndex):
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff <= self.epsilon:
            actionIndex = enviroment.actionspace_sample(actionIndexes)
            return actionIndex
        
        
        q_values = self.q_network.predict(stateIndex)
        seen = []
        for x in range(len(q_values[0])):                             
            if x in actionIndexes[0] or x in actionIndexes[1]:
                seen.append(1)
            else:
                seen.append(0)

        a = ma.array(q_values[0], mask=seen)
        return np.argmax(a)

    def predict(self, actionIndexes, stateIndex):
        q_values = self.q_network.predict(stateIndex)
        seen = []
        for x in range(len(q_values[0])):                             
            if x in actionIndexes[0] or x in actionIndexes[1]:
                seen.append(1)
            else:
                seen.append(0)

        a = ma.array(q_values[0], mask=seen)
        return np.argmax(a)


    def decay(self, totalSteps):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*totalSteps) 

