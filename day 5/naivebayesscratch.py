import csv
import random
import numpy as np
import math

#Handle Data
def loadCsv(filename):
	with open(filename, 'r') as f:
		lines = csv.reader(f)
		dataset = list(lines)
		for i in range(len(dataset)):
			dataset[i]=[float(x) for x in dataset[i]]
	dataset = np.array(dataset)
	return dataset

#split Dataset
def splitDataset(dataset, splitRatio):
	s = int(len(dataset)*splitRatio)
	np.random.shuffle(dataset)
	train_set, test_set = dataset[:s, :], dataset[s:, :]
	return [train_set, test_set]

##summarize Data

#separate Data by class
def separateByClass(dataset):
	separated = {}
	j=0
	for i in dataset[:,-1]:
		if i not in separated :
			separated[i]=[]
		separated[i].append(list(dataset[j]))
		j+=1
	return separated

#calculate mean
def mean(dataset, class_id):
	return np.mean(dataset[:,class_id])

#calculate standard deviation
def stdev(dataset, class_id):
	return np.std(dataset[:,class_id])

#summarize attributes by class
def summarize(dataset):
	dataset = np.array(dataset)
	summaries = [(mean(dataset, attribute), stdev(dataset, attribute)) for attribute in range(np.size(dataset,1))]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	summary = {}
	separated = separateByClass(dataset)
	for classValue, instances in separated.items():
		summary[classValue] = summarize(instances)
	return summary

#calculate gaussian probability
def GaussianNB(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2))/(2*math.pow(stdev,2)))
	return (1/(math.sqrt(2*math.pi))*stdev)*exponent

#calculate class probabilities
def calculateProbability(summary, inputVector):
	prob = {}
	for classKey, value in summary.items():
		prob[classKey] = 1
		for i in range(len(value)):
			mean, stdev = value[i]
			x = inputVector[i]
			prob[classKey] *= GaussianNB(x,mean,stdev)
	return prob

#make a prediction
def predict(summary, inputVector):
	probabilities = calculateProbability(summary,inputVector)
	bestLabel, bestProb = None, -1
	for k,v in probabilities.items():
		if bestLabel is None or v > bestProb:
			bestProb = v
			bestLabel = k
	return bestLabel

#make predictions
def getPredictions(summary, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summary, testSet[i])
		predictions.append(result)
	return predictions

def accuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x,-1] == predictions[x]:
			correct+=1
	return (correct/float(len(testSet)))*100.0

def inputfile():
	filename = input('Input name of csv file: ')
	return filename

def main():
	filename = inputfile()+'.csv'
	dataset = loadCsv(filename)
	print('Loaded data file',filename, 'with',len(dataset), 'rows')
	train_set, test_set = splitDataset(dataset, 0.7)
	print('train',len(train_set), ' test', len(test_set))
	#prepare model
	summary = summarizeByClass(train_set)
	#test model
	predictions = getPredictions(summary,test_set)
	acc = accuracy(test_set, predictions)
	print("Accuracy: ",acc)

main()