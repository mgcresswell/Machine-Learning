import MovieRecommender_QLearning as movieEnv
import pandas as pd


#Similar to a grid search 
#Used to call recommder agents 
#preform optimization
epsilon = 1.0                 
max_epsilon = 1.0             
min_epsilon = 0.01          
decay_rate = .07
#actions = 
data = []
numberOfUsers = [10,25,50,75,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
decayRates = [.0001,.001,.005,.01,.02,.03,.04,.05,.07,.1,.2,.3,.5,1]
states = [[2858,260,1196,1210,480,2028], [2858,260,1196,1210,480,2028,589],[2858,260,1196,1210,480,2028,589],[2858,260,1196,1210,480,2028,589,2571,1270],[2858,260,1196,1210,480,2028,589,2571,1270,593]]
t = len(states)
#for i in range(len(states)):
for x in range(5):
   totalTime, successPercent, availableTrainingUsers, availableTestUsers = movieEnv.ExecuteMovieRecommender([2858,260,1196,1210,480], epsilon, max_epsilon, min_epsilon, decay_rate, 1000)
   newRow = [x,totalTime, successPercent, availableTrainingUsers, availableTestUsers]
   data.append(newRow)
   print(newRow)

data = pd.DataFrame(data= data, columns=['State','Time','Success','TrainingUsers','TestUsers'])
data.to_csv('C:\\Users\\cressm\\Desktop\\TCSS 556\\StateOutput.csv')

