import pandas as pd
import itertools
from itertools import combinations 
import numpy as np
import random
import sys
import numpy.ma as ma
from sklearn.utils import shuffle

#Object to store user's movie ratings
class UserMovie:
  def __init__(self, rating, movieId, movieIdIndex):
    self.rating = rating
    self.movieId = movieId
    self.movieIdIndex = movieIdIndex


class MovieEnviroment: 
    def __init__(self, actions, totalUsers):
        self.actions = actions
        self.totalTrainingUsers = totalUsers
        self.states = self.generateStates()
        self.trainDf, self.testDf = self.getData()
        self.availableTrainingUsers = self.trainDf.shape[0]
        self.availableTestUsers = self.testDf.shape[0]

       
    def createQTable(self):
        qtable = np.zeros((self.getStateSpaceSize(), self.getActionSpaceSize()))
        return qtable

    #create all subsets of the set of n actions
    def generateStates(self): 
        states = []
        states.append([])
        for x in range(1,len(self.actions)):
            comb = combinations(self.actions, x)  
            for i in list(comb):  
                i = np.sort(i)  
                states.append(i)  
        return states

    def splitData(self, data):       
        splitVal = data.shape[0] * .7
        splitVal = round(splitVal)
        trainDf = data.iloc[:splitVal,:] 
        testDf = data.iloc[splitVal:,:] 
        return trainDf, testDf

    #read data from dataset
    #replace string here with data file
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
        userDf.set_index('UserID')
        userDf = shuffle(userDf)
        
        trainDf, testDf = self.splitData(userDf)
        testDiff = trainDf.iloc[self.totalTrainingUsers:,:]
        trainDf = trainDf.iloc[:self.totalTrainingUsers,:]
        testDf = pd.concat([testDf, testDiff])
        return trainDf, testDf

    def getStateIndex(self, state):
        i = 0
        stateValues = []
        for x in range(len(state)):
            stateValues.append(self.getActionByIndex(state[x]))

        state = np.sort(stateValues)  
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
        notWatched = []
        for i in range(len(self.actions)):
            if i not in state[0] and i not in state[1]:
                notWatched.append(i)
        return random.choice(notWatched)


    def getQTableAction(self, curUserStateIndexes, row):        
        seen = []
        for x in range(len(row)):                             
            if x in curUserStateIndexes[0] or x in curUserStateIndexes[1]:
                seen.append(1)
            else:
                seen.append(0)

        a = ma.array(row, mask=seen)
        actionIndex = np.argmax(a)
        return actionIndex

    #create data structure for user ratings
    def setUserState(self, user, step):
        curActionIndexes = []
        likedIndexes = []
        seenIndexes = []
        state = self.getStateByIndex(step)
        if len(state) == 0:
            curActionIndexes.append(likedIndexes)
            curActionIndexes.append(seenIndexes)      
            return curActionIndexes
        else:
            newState = False
            for x in range(len(state)):
                for y in range(len(user)):
                    if state[x] == user[y].movieId:
                        if user[y].rating >= 4:                       
                            likedIndexes.append(user[y].movieIdIndex)
                        else:
                            seenIndexes.append(user[y].movieIdIndex)

            curActionIndexes.append(likedIndexes)
            curActionIndexes.append(seenIndexes)      
            return curActionIndexes

    def getTrainingUserWithState(self, userIndex):
        userRow = self.trainDf.iloc[userIndex]
        movies = userRow.drop(['UserID']).to_numpy()
        state = []
        for i in range(len(movies)):
            movie = UserMovie(movies[i], self.getActionByIndex(i),i)
            state.append(movie)
        return state

    def getTestUserWithState(self, userIndex):
        userRow = self.testDf.iloc[userIndex]
        movies = userRow.drop(['UserID']).to_numpy()
        state = []
        for i in range(len(movies)):
            movie = UserMovie(movies[i], self.getActionByIndex(i),i)
            state.append(movie)
        return state

    def getAvailableTrainingUsers(self):
        return self.availableTrainingUsers

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

    
    def step(self, curStateIndexes, actionIndex, userState):
        unseenMovies = [] 
        actionRating = None
        #get unseen movies, ie movie not in current state
        for x in range(len(userState)):
               if userState[x].movieIdIndex != actionIndex and userState[x].movieIdIndex not in curStateIndexes[0] and userState[x].movieIdIndex not in curStateIndexes[1]:
                   unseenMovies.append(userState[x].rating)
               elif userState[x].movieIdIndex == actionIndex:
                   actionRating = userState[x].rating
        #get highest rating of unseen movies        
        if len(unseenMovies) > 0:
            maxRating = np.max(unseenMovies)
        else:
            maxRating = 0

        #set state of metric
        metric = 0
        if actionRating >= maxRating:
            metric = 1
        
        #append action to appropriate array 
        #based on actual user rating
        if actionRating >= 4:
            curStateIndexes[0].append(actionIndex)
        else:
            curStateIndexes[1].append(actionIndex)
                       
        if (len(curStateIndexes[0]) + len(curStateIndexes[1]) >= len(self.actions)):  
            return curStateIndexes, actionRating, maxRating, metric, True
        else:
            return curStateIndexes, actionRating, maxRating, metric, False
            





        




