import csv
import os
import pandas as pd
import numpy as np

# 1 - my
# 2 - 570
# 3 - 567
# 4 - 540
# 5 - 563

file = pd.read_csv("ID596 - training - final.csv")

csvfile = open("aGlucoseScore.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)
csvfile_writer.writerow(["risk score"])

for i in range(0,len(file)):
    if file["glucose"][i] == 0:
        riskScore = file["finger stick"][i] * (1+0*int(float(file["exercise"][i]))+0*file["illness"][i]+0*file["stressors"][i]-0.1*file["sleep"][i])/100
        csvfile_writer.writerow([riskScore])

    else:
        riskScore = file["glucose"][i] * (1+0*int(float(file["exercise"][i]))+0*file["illness"][i]+0*file["stressors"][i]-0.1*file["sleep"][i])/100
       
        csvfile_writer.writerow([riskScore])

