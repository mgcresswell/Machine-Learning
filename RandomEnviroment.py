import pandas as pd
import itertools
from itertools import combinations 
import numpy as np
import random
import sys
import numpy.ma as ma
from sklearn.utils import shuffle

#Enviorment similar to other enviroment but doesnt split data

class UserMovie:
  def __init__(self, rating, movieId, movieIdIndex):
    self.rating = rating
    self.movieId = movieId
    self.movieIdIndex = movieIdIndex

class ActionMovie:
  def __init__(self, action, actionId):
    self.action = actionId
    self.actionId = actionId

class RandomMovieEnviroment: 
    def __init__(self, actions, totalUsers):
        self.actions = actions
        self.totalUsers = totalUsers
        self.states = self.generateStates()
        self.trainDf = self.getData()
        self.keepTraining = True
        self.availableTrainingUsers = self.trainDf.shape[0]
        self.currentAvailableTrainingUsers = 0

       
    def createQTable(self):
        qtable = np.zeros((self.getStateSpaceSize(), self.getActionSpaceSize()))
        return qtable

    def generateStates(self): 
        states = []
        states.append([])
        for x in range(1,len(self.actions)):
            comb = combinations(self.actions, x)  
            for i in list(comb):  
                i = np.sort(i)  
                states.append(i)  
        return states


    def getData(self):
        df = pd.read_csv('C:\\Users\\cressm\\Desktop\\TCSS 556\\ml-1m\\ratings.dat', header=None, sep='::', engine='python')
        df.columns = ['UserID','MovieID','Rating','Timestamp']        
        userDf = pd.DataFrame() 
        for x in range(len(self.actions)):
            if x == 0:
                userDf = df[df['MovieID'] == self.actions[x]]
                userDf = userDf[['UserID','Rating']]
            else:
                temp = df[df['MovieID'] == self.actions[x]]
                temp = temp[['UserID','Rating']]
                userDf = pd.merge(userDf, temp, on="UserID")
             
        userDf.columns = self.getColumnNames()
        userDf['Stepped'] = 0
        userDf.set_index('UserID')
        return userDf

    def resetTraining(self):
        self.keepTraining = True
        self.trainDf['Stepped'] = 0
        self.currentAvailableTrainingUsers = self.availableTrainingUsers

    def resetTesting(self):
        self.keepTesting = True
        self.testDf['Stepped'] = 0
        self.currentAvailableTestUsers = self.availableTestUsers

    def getStateIndex(self, state):
        i = 0
        state = np.sort(state)  
        for x in self.states:
            if np.array_equal(state,x):
                return i
            else:
                i = i + 1

        return None

    def getActionIndex(self, action):
        i = 0
        for x in self.actions:
            if action == x:
                return i
            else:
                i = i + 1

    def actionspace_sample(self, state):
        lst3 = [value for value in self.actions if value not in state[0]] 
        lst4 = [value for value in lst3 if value not in state[1]] 
        return random.choice(lst4)

    def getTrainingUser(self):
        stepTable = self.trainDf[self.trainDf['Stepped'] == 0]
        userId = stepTable.iloc[0]['UserID'] 
        return userId

    def getQTableAction(self, cur_state_index, curUserStateIndexes, row):        
        seen = []
        for x in range(len(row)):                             
            if x in curUserStateIndexes[0] or x in curUserStateIndexes[1]:
                seen.append(1)
            else:
                seen.append(0)

        a = ma.array(row, mask=seen)
        actionIndex = np.argmax(a)
        action = self.actions[actionIndex]
        return action, actionIndex

    def setUserState(self, user, step):
        curActionValues = []
        curActionIndexes = []
        likedValues = []
        likedIndexes = []
        seenValues = []
        seenIndexes = []
        state = self.getStateByIndex(step)
        if len(state) == 0:
            curActionValues.append(likedValues)
            curActionValues.append(seenValues)
            curActionIndexes.append(likedIndexes)
            curActionIndexes.append(seenIndexes)      
            return curActionValues, curActionIndexes
        else:
            newState = False
            for x in range(len(state)):
                for y in range(len(user)):
                    if state[x] == user[y].movieId:
                        if user[y].rating >= 4:                       
                            likedValues.append(user[y].movieId)
                            likedIndexes.append(user[y].movieIdIndex)
                        else:
                            seenValues.append(user[y].movieId)
                            seenIndexes.append(user[y].movieIdIndex)

            curActionValues.append(likedValues)
            curActionValues.append(seenValues)
            curActionIndexes.append(likedIndexes)
            curActionIndexes.append(seenIndexes)      
            return curActionValues, curActionIndexes

    def getTrainingUserWithState(self):
        stepTable = self.trainDf[self.trainDf['Stepped'] == 0]
        userId = stepTable.iloc[0]['UserID'] 
        movieRow = stepTable.iloc[0]
        movies = movieRow.drop(['UserID','Stepped']).to_numpy()
        state = []
        for i in range(len(movies)):
            movie = UserMovie(movies[i], self.getActionByIndex(i),i)
            state.append(movie)

        if stepTable.shape[0] == 1:
            self.keepTraining = False;
        else:
            userIdIndex = self.trainDf.index[self.trainDf['UserID'] == userId].tolist()
            self.trainDf['Stepped'][userIdIndex[0]] = 1
        return state

    def getTestUserWithState(self):
        stepTable = self.testDf[self.testDf['Stepped'] == 0]
        userId = stepTable.iloc[0]['UserID'] 
        movieRow = stepTable.iloc[0]
        movies = movieRow.drop(['UserID','Stepped']).to_numpy()
        state = []
        for i in range(len(movies)):
            movie = UserMovie(movies[i], self.getActionByIndex(i),i)
            state.append(movie)

        if stepTable.shape[0] == 1:
            self.keepTesting = False;
        else:
            userIdIndex = self.testDf.index[self.testDf['UserID'] == userId].tolist()
            self.testDf['Stepped'][userIdIndex[0]] = 1
        return state

    def getAvailableTrainingUsers(self):
        return self.availableTrainingUsers

    def getTestUser(self):
        stepTable = self.testDf[self.testDf['Stepped'] == 0]
        userId = stepTable.iloc[0]['UserID'] 
        return userId

    def getActionSpaceSize(self):
        return len(self.actions)

    def getStateSpaceSize(self):
        return len(self.states)

    def getStateByIndex(self, index):
        return self.states[index]

    def getActionByIndex(self, index):
        return self.actions[index]

    def getColumnNames(self):
        names = ['UserID'] 
        for x in range(len(self.actions)):
            names.append(self.actions[x])
        return names

    def trainStep(self, curStateValues, curStateIndexes, action, userState):
        if action in curStateValues[0]:
            return state, -5, userId, False
        else:
            rating = None
            for x in range(len(userState)):
               if userState[x].movieId == action:
                   rating = userState[x].rating
                     
            if rating >= 4:
                curStateValues[0] = np.append(curStateValues[0], action)
                curStateIndexes[0] = np.append(curStateIndexes[0], self.getActionIndex(action))
            else:
                curStateValues[1] = np.append(curStateValues[1], action)
                curStateIndexes[1] = np.append(curStateIndexes[1], self.getActionIndex(action))

            if (len(curStateValues[0]) + len(curStateValues[1]) >= len(self.actions)): 
                return curStateValues, curStateIndexes, rating, True
            else:
                return curStateValues, curStateIndexes, rating, False
            
    
    def testStep(self, curStateValues, curStateIndexes, action, userState):        
        unseenMovies = [] 
        actionRating = None
        for x in range(len(userState)):
               if userState[x].movieId != action and userState[x].movieId not in curStateValues[0] and userState[x].movieId not in curStateValues[1]:
                   unseenMovies.append(userState[x].rating)
               elif userState[x].movieId == action:
                   actionRating = userState[x].rating
                   
        if len(unseenMovies) > 0:
            maxRating = np.max(unseenMovies)
        else:
            maxRating = 0

        metric = 0
        if actionRating >= maxRating:
            metric = 1
                 
        if actionRating >= 4:
            curStateValues[0] = np.append(curStateValues[0], action)
            curStateIndexes[0] = np.append(curStateIndexes[0], self.getActionIndex(action))
        else:
            curStateValues[1] = np.append(curStateValues[1], action)
            curStateIndexes[1] = np.append(curStateIndexes[1], self.getActionIndex(action))
                       
        if (len(curStateValues[0]) + len(curStateValues[1]) >= len(self.actions)):  
            return curStateValues, curStateIndexes, actionRating, maxRating, metric, True
        else:
            return curStateValues, curStateIndexes, actionRating, maxRating, metric, False





        





