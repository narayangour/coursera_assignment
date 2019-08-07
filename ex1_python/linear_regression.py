import sklearn			# Library useful for different dataset and classifier
import numpy as np		# Array
import matplotlib.pyplot as plt	# For plotting graph of data
import csv			# CSV related 
import pandas as pd 		# Load the Pandas libraries with alias 'pd' 
import itertools  
import sys
import time
import progressbar
import datetime


start_time = time.time()

# convert and copy data from ex1data1.txt to ex1data1.csv
def ConvTxtToCsv(filename):
	with open(filename, 'r') as in_file:
		stripped = (line.strip() for line in in_file)
		lines = (line.split(",") for line in stripped if line)
		with open('ex1data1.csv', 'w') as out_file:
			writer = csv.writer(out_file)
			writer.writerows(lines)
	return

# list data from CSV file
def readMyCsvFile(filename):
    X = []
    Y = []
    count=0
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
           if(count==0):
              count = count+1
              continue
           X.append(row[0])
           Y.append(row[1])
 
    return X, Y

def cost_fuction(inp, out, theta_0=0, theta_1=0.5):
	total = 0
	for (a, b) in zip(inp, out):
		total = total + (((theta_0+theta_1*a)-b)**2)
	return float(total/2*len(inp))

def gradient_descent(inp, out, theta_0=0, theta_1=0.5, epochs = 10,  learning_rate=0.000000000001):
	SumForThetaZero = 0
	SumForThetaOne = 0
	error_list = []
	epoch_list = []
	print '\nInput epochs - :',epochs, "And learning_rate - :",learning_rate,"\n" 
	with open('output.csv', 'rw') as readFile:
		reader = csv.reader(readFile)
		lines = list(reader)
	bar = progressbar.ProgressBar(maxval=epochs,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	for i in range(epochs):
		bar.update(i+1)
		for (a, b) in zip(inp, out):
			SumForThetaZero = SumForThetaZero + (theta_0+theta_1*a)-b
			SumForThetaOne = SumForThetaOne + ((theta_0+theta_1*a)-b)*a
		theta_0 = theta_0 - ((SumForThetaZero*learning_rate)/len(inp))
		theta_1 = theta_1 - ((SumForThetaOne*learning_rate)/len(inp))
		cost = cost_fuction(inp, out, theta_0, theta_1)
		if(i==0):
			error_list.append(cost)
			epoch_list.append(i)
			temp = cost
		elif(cost<temp):
			error_list.append(cost)
			epoch_list.append(i)
			temp = cost
	bar.finish()
	lines[1][1] = theta_0
	lines[2][1] = theta_1
	lines[3][1] = str(epochs)
	lines[4][1] = str(temp)
	lines[5][1] = str(learning_rate)
	print"\nFinal weight Values after %d epochs"%(epochs)
	print "theta_0 -> ",theta_0
	print "theta_1 -> ",theta_1
	print "cost    -> ",temp,"\n\nAnd updated to output.csv file in current directory..!"
	with open('output.csv', 'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerows(lines)
	return theta_0,theta_1,temp,error_list,epoch_list


if(len(sys.argv)!=3):
	print """\
	Please enter valid input -
	Standard Argument : python filename.py epochs learning_rate
	Ex:- python test.py 100 .01
	Exiting from script"""
	sys.exit(0)

ConvTxtToCsv("ex1data1.txt")
data = pd.read_csv("ex1data1.csv",usecols=['x', 'y'])
x = data.x.tolist()
y = data.y.tolist()

print "Number of observation is ",len(x),"in dataset"

#plotting input data
plt.plot(x, y);
plt.title('input_data')
plt.xlabel('input');
plt.ylabel('output');
plt.show()


# Compute cost 
cost = cost_fuction(x,y)


# Gradient descent 
theta_0,theta_1,temp,error_list ,epoch_list= gradient_descent(x, y, epochs = int(sys.argv[1]), learning_rate = float(sys.argv[2]))
print"\nTraining Done..!!\n"

# Testing 
inp_val = input("Enter number to predict respective value - ")
output = theta_0+theta_1*(float(inp_val))
print"Predicted output 			 -",output

fig = plt.figure()
plt.plot(error_list,epoch_list);
plt.title('Gradient_Descent(epoch_vs_error)')
plt.xlabel('error');
plt.ylabel('epoch');
#plt.show()
plt.draw()
plt.pause(1) # <-------
raw_input("Hit Enter To Close..Graph!!")
plt.close(fig)

print"\nPragram End ..			         - %s" % str(datetime.timedelta(seconds=time.time() - start_time)),"HH:MIN:SEC:MS\n"














