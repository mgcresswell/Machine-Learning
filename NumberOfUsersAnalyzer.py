import numpy as np
import matplotlib.pyplot as plt
import pymysql
import pymysql.cursors
import pandas as pd
from matplotlib.pyplot import figure

figure(num=None, figsize=(14, 6), dpi=200, facecolor='w', edgecolor='k')
df = pd.read_csv('C:\\Users\\cressm\\Desktop\\TCSS 556\\NumOfUsers_Average.csv')
  


qlearning = df.iloc[0]
deepQlearning = df.iloc[1]
x = [10,25,50,75,100,150,200,250,300,350,400,450,500,550,600,650]

plt.plot(qlearning)
plt.plot(deepQlearning)
plt.title('Recommendation Success vs Number of Users')
plt.xlabel('Number of Users')
plt.ylabel('Recommendation Success')
plt.xticks(np.arange(len(x)), x, rotation=20)
plt.legend(["Q-Learning", "Deep Q-Learning"], loc ="lower right") 
plt.savefig('NumberOfUsers.png')
plt.show()


