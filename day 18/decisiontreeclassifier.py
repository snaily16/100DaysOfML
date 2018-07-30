#dataset
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

#column labels
header = ['color','diameter','label']

#find the unique values for a column in a dataset
def unique_vals(rows, col):
	return set([row[col] for row in rows])

#counts the number of each type of example in a dataset
def class_counts(rows):
	counts = {}
	for row in rows:
		#label is the last column
		label = row[-1]
		if label not in counts:
			counts[label] = 0
		counts[label] += 1
	return counts

#test if value is numeric
def is_numeric(value):
	return isinstance(value, int) or isinstance(value, float)

#question to partition a dataset
class Question:
	#records column number and column value
	def __init__(self, column, value):
		self.column = column
		self.value = value

	#compare feature value in an example to feature value stored in the question
	def match(self, example):
		val = example[self.column]
		if is_numeric(val):
			return val >= self.value
		else:
			return val == self.value

	#helper method to print question in readable format
	def __repr__(self):
		condition = "=="
		if is_numeric(self.value):
			condition = ">="
		return "Is %s %s %s?" % (header[self.column], condition, str(self.value))

#partition a dataset
#for every row, check if it matches the question
def partition(rows, question):
	true_rows, false_rows =[], []
	for row in rows:
		if question.match(row):
			true_rows.append(row)
		else:
			false_rows.append(row)
	return true_rows, false_rows


#calculate Gini impurity for list of rows
def gini(rows):
	counts = class_counts(rows)
	impurity = 1
	for lbl in counts:
		prob_of_lbl = counts[lbl] / float(len(rows))
		impurity -= prob_of_lbl ** 2
	return impurity

#Information Gain
#uncertainty of the starting node minus the weighted impurity of two child nodes
def info_gain(left, right, current_uncertainty):
	p = float(len(left)) / (len(left) + len(right))
	return current_uncertainty - p * gini(left) - (1-p) * gini(right)

#find the best question to ask by iterating over every feature / value and calculate the information Gain
def find_best_split(rows):
	best_gain = 0	#keep track of best info gain
	best_question = None
	current_uncertainty = gini(rows)
	n_features = len(rows[0]) -1 #number of columns

	for col in range(n_features):
		values = set([row[col] for row in rows])	#unique values in the column
		for val in values:	#for each value
			question = Question(col, val)
			#split dataset
			true_rows, false_rows = partition(rows, question)

			#skip split if doesn't divide the dataset
			if len(true_rows) == 0 or len(false_rows) == 0 :
				continue

			#calculate the information gain from this split
			gain = info_gain(true_rows, false_rows, current_uncertainty)

			if gain >= best_gain:
				best_gain, best_question = gain, question

	return best_gain, best_question


#Leaf node classifies data
#holds a dictionary of class, number of times it appears in the rows
class Leaf:
	def __init__(self, rows):
		self.predictions = class_counts(rows)


#Decision Node asks a question
#holds a reference to question, and to the two child nodes
class Decision_Node:
	def __init__(self, question, true_branch, false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch


#Builds the tree
#Rules of recursion:	1. Believe that it works
#						2. Start by checking for the base case (no further information gain)
#						3. Prepare for giant stack traces	
def build_tree(rows):
	#partition dataset on each of unique attribute
	#calculate gain and return question that produces highest gain
	gain, question = find_best_split(rows)

	#best case : no further info gain, return a Leaf
	if gain == 0:
		return Leaf(rows)

	#a useful feature to partition on
	true_rows, false_rows = partition(rows, question)

	#recursively build true and false branch
	true_branch = build_tree(true_rows)
	false_branch = build_tree(false_rows)

	#return a question node
	return Decision_Node(question, true_branch, false_branch)


#print tree
def print_tree(node, spacing = ""):
	#Base case: we've reached a Leaf
	if isinstance(node, Leaf):
		print(spacing + "Predict", node.predictions)
		return
	#print question at this node
	print(spacing + str(node.question))

	#call this function recursively on the true and false branch
	print(spacing + '--> True:')
	print_tree(node.true_branch, spacing + " ")

	print(spacing + '--> False:')
	print_tree(node.false_branch, spacing + " ")

def classify(row, node):
	#leaf node
	if isinstance(node, Leaf):
		return node.predictions

	#decide whether to follow true_branch or false_branch
	if node.question.match(row):
		return classify(row, node.true_branch)
	else:
		return classify(row, node.false_branch)

#print the predictions at a leaf
def print_leaf(counts):
	total = sum(counts.values()) * 1.0
	probs = {}
	for lbl in counts.keys():
		probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
	return probs


if __name__ == '__main__':
	my_tree = build_tree(training_data)
	print_tree(my_tree)

	#evaluate
	testing_data = [
	    ['Green', 3, 'Apple'],
	    ['Yellow', 4, 'Apple'],
	    ['Red', 2, 'Grape'],
	    ['Red', 1, 'Grape'],
	    ['Yellow', 3, 'Lemon'],
	]

	for row in testing_data:
		print("Actual: ", row[-1], " Predicted: ", print_leaf(classify(row, my_tree)))