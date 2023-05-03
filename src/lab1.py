
import os
import csv
from matplotlib import pyplot as plt

os.chdir(r"C:\PycharmProjects\Data_Science_and_Analysis\data")

temp_list = []
mag_list = []

with open('StarTypeDataset.csv', newline='') as csvfile:
    spam_reader = csv.reader(csvfile, delimiter=',')
    next(spam_reader)
    for row in spam_reader:
        temp_list.append(float(row[0]))
        mag_list.append(float(row[1]))

print(temp_list[0])
plt.scatter(temp_list, mag_list)
plt.xlabel("Temperature")
plt.ylabel("Absolute magnitude")
plt.show()
