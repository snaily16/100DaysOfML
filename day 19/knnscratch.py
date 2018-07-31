#KNN
import csv
import random
import math
import operator

#Handle Data
def loadDataset(filename, split, trainingSet=[], testSet=[]):

	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)

		for x in range(len(dataset)-1):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			#split dataset
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])

#Similarity
#Calculate Euclidean distance measure
def euclideanDistance(x,y, length):
	distance = 0
	for i in range(length):
		distance += pow((x[i] - y[i]),2)
	return math.sqrt(distance)

#Neighbors
#locate k most similar data instances
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance) -1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

#Response based on the neighbors
#getting the majority voted response from a number of neighbors
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes [response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

#Accuracy of predictions
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

if __name__ == "__main__":

	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.csv', split, trainingSet, testSet)
	print('Train: ', len(trainingSet))
	print('Test: ', len(testSet))

	#generate predictions
	predictions = []
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print("Predicted: ", result, " Actual: ", testSet[x][-1])
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ',accuracy)