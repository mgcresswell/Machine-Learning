#Mike Cresswell
#TCSS 555
#Data Cleaning

import os
import numpy
import csv
import pandas as pd
import numpy as np

#Loop through files and directories of unzipped data folder
#Read data from file names and directory names
#Add them as features to the dataset

#place zip file in current working dir
def readTranscripts():
    path = os.getcwd()
    rootdir = path+"\\op_spam_v1.4\\"
    records = []
    id = 1
    for subdir, dirs, files in os.walk(rootdir):
         for file in files:
             subArray = subdir.split("\\")
             polarity = subArray[6].split("_")[0];
             deceptive = subArray[7].split("_")[0];
             source = subArray[7].split("_")[2];
             hotel = file.split("_")[1]
             fullFileName = subdir + "\\" +file
             with open(fullFileName, "r") as f:
                text = f.read()
                d = {'id': id, 'deceptive': deceptive, 'hotel':hotel, 'polarity':polarity,'source':source,'text': text}
                records.append(d)
             id = id + 1
    df = pd.DataFrame(data=records, columns=['id', 'deceptive','hotel','polarity','source','text'])
    return df

#Driver code
df = readTranscripts();

df.to_csv(path+'\\deceptive-opinion.csv',index=False)




