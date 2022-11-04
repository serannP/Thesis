import os
import pandas as pd
import numpy as np
import csv

file = pd.read_csv("ID591 - training - final.csv")

csvfile = open("aClassifyRisk.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)
csvfile_writer.writerow(["classify risk"])

for i in range(0,len(file)):
    if (file["risk score"][i] > 1.4):
        classify = 1
        csvfile_writer.writerow([classify])
    elif (file["risk score"][i] < 0.9):
        classify = -1
        csvfile_writer.writerow([classify])
    else:
        classify = 0
        csvfile_writer.writerow([classify])

