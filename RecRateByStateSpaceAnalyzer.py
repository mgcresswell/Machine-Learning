import numpy as np
import matplotlib.pyplot as plt
import pymysql
import pymysql.cursors
import pandas as pd
from matplotlib.pyplot import figure

figure(num=None, figsize=(14, 6), dpi=200, facecolor='w', edgecolor='k')

qlearning = [0.841384,0.804202,0.81179,0.755434,0.74943]
deepQlearning = [0.85634,0.815562,0.82187,0.771172,0.76265]
random = [.776,.744,.744,.7011,.6892] 
x = [64,128,256,512,1024]

plt.plot(qlearning)
plt.plot(deepQlearning)
plt.plot(random)
plt.title('Recommendation Success vs Action Space Size')
plt.xlabel('Action Space Size')
plt.ylabel('Recommendation Success')
plt.xticks(np.arange(len(x)), x, rotation=20)
plt.legend(["Q-Learning", "Deep Q-Learning", "Random"], loc ="lower right") 
plt.savefig('RecRateByStateSpace.png')
plt.show()
