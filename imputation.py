import os
import pandas as pd
import numpy as np
import csv

file = pd.read_csv("ID552 - training.csv")

csvfile = open("aglucose.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["glucose"]].replace(0, np.nan )

file2 = file2[["glucose"]].fillna(method="ffill", limit = 5)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("afinger.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["finger stick"]].replace(0, np.nan )

file2 = file2[["finger stick"]].fillna(method="ffill", limit = 5)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("abolus.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["bolus amount"]].replace(0, np.nan )

file2 = file2[["bolus amount"]].fillna(method="bfill", limit = 2)
file2 = file2[["bolus amount"]].fillna(method="ffill", limit = 2)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

file2 = file[["meal type"]].replace(0, np.nan )

csvfile = open("ameal.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file2[["meal type"]].fillna(method="bfill", limit = 2)
file2 = file2[["meal type"]].fillna(method="ffill", limit = 2)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("acarb.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["carb amount"]].replace(0, np.nan )

file2 = file2[["carb amount"]].fillna(method="bfill", limit = 2)
file2 = file2[["carb amount"]].fillna(method="ffill", limit = 2)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("astress.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["stressors"]].replace(0, np.nan )

file2 = file2[["stressors"]].fillna(method="bfill", limit = 2)
file2 = file2[["stressors"]].fillna(method="ffill", limit = 2)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("ahypo.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["hypo"]].replace(0, np.nan )

file2 = file2[["hypo"]].fillna(method="bfill", limit = 2)
file2 = file2[["hypo"]].fillna(method="ffill", limit = 2)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("aillness.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["illness"]].replace(0, np.nan )

file2 = file2[["illness"]].fillna(method="bfill", limit = 2)
file2 = file2[["illness"]].fillna(method="ffill", limit = 2)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("aheart.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["heart rate"]].replace(0, np.nan )

file2 = file2[["heart rate"]].fillna(method="ffill", limit = 5)

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))

csvfile = open("abasal.csv",'w',encoding='utf-8', newline = '')
csvfile_writer = csv.writer(csvfile)

file2 = file[["basal"]].replace(0, np.nan )

file2 = file2[["basal"]].fillna(method="ffill")

for row in file2:
    csvfile_writer.writerows(zip(file2[row]))
