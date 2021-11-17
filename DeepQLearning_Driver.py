import MovieRecommender_DeepQLearning as movieEnv
import pandas as pd


#Driver to call Deep Q Driver
epsilon = 1.0                 
max_epsilon = 1.0             
min_epsilon = 0.01          
decay_rate = .1
data = []
numberOfUsers = [10,25,50,75,100,200,300,400,500,600,800,900,1000]
decayRates = [.0001,.001,.005,.01,.02,.03,.04,.05,.07,.1,.2,.3,.5,1]

for i in range(len(numberOfUsers)):
    for x in range(5):
        totalTime, successPercent, availableTrainingUsers, availableTestUsers = movieEnv.ExecuteMovieRecommender([2858,260,1196,1210,480], epsilon, max_epsilon, min_epsilon, decay_rate, 50)
        newRow = [totalTime, successPercent, availableTrainingUsers, availableTestUsers]
        data.append(newRow)
        print(newRow)
#data = pd.DataFrame(data= data, columns=['Time','Success','TrainingUsers','TestUsers'])
#data.to_csv('C:\\Users\\cressm\\Desktop\\TCSS 556\\DeepQ_NumOfUserResults.csv')
