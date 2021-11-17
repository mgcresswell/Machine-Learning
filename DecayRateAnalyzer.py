import numpy as np
import matplotlib.pyplot as plt
import pymysql
import pymysql.cursors
import pandas as pd
from matplotlib.pyplot import figure

figure(num=None, figsize=(14, 6), dpi=200, facecolor='w', edgecolor='k')
df = pd.read_csv('C:\\Users\\cressm\\Desktop\\TCSS 556\\DecayRateResults.csv')
  
qlearning = df.iloc[0]
deepQlearning = df.iloc[1]
x = [0.0001,0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.2,0.3,0.5,1]

plt.plot(qlearning)
plt.plot(deepQlearning)
plt.title('Recommendation Success vs Decay Rate')
plt.xlabel('Decay Rate')
plt.ylabel('Recommendation Success')
plt.xticks(np.arange(len(x)), x, rotation=20)
plt.legend(["Q-Learning", "Deep Q-Learning"], loc ="lower right") 
plt.savefig('DecayRateGraph.png')
plt.show()