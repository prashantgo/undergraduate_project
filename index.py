import pandas as pd

pairs = dict()

def restrictive_training():
	global pairs
	d1 = pd.read_table("News_Train.tsv", header=None)
	d2 = pd.read_table("WikiNews_Train.tsv", header=None)
	d3 = pd.read_table("Wikipedia_Train.tsv", header=None)
	frames = [d1, d2, d3]
	table = pd.concat(frames)
	del d1
	del d2
	del d3

	for i, j, k in zip(table[4], table[7], table[8]):
		if i in pairs:
			pass
		else:
			if j >= 1 or k >= 1:
				pairs[i] = 1
			elif len(i) >= 10:
				pairs[i] = 1
			else:
				pairs[i] = 0

def baseline_training():
	global pairs
	d1 = pd.read_table("News_Train.tsv", header=None)
	d2 = pd.read_table("WikiNews_Train.tsv", header=None)
	d3 = pd.read_table("Wikipedia_Train.tsv", header=None)
	frames = [d1, d2, d3]
	table = pd.concat(frames)
	del d1
	del d2
	del d3
	print(len(table))

	for i, j in zip(table[4], table[9]):
		if i in pairs:
			if j == 1:
				pairs[i].append(j)
			elif len(i) >= 10:
				pairs[i].append(1)
			else:
				pairs[i].append(0)
		else:
			if j == 1:
				pairs[i] = [1]
			elif len(i) >= 10:
				pairs[i] = [1]
			else:
				pairs[i] = [0]

	for i in pairs.keys():
		if 1 in pairs[i]:
			pairs[i] = 1
		else:
			pairs[i] = 0
			
def baseline_with_no_length_part_training():
	global pairs
	d1 = pd.read_table("News_Train.tsv", header=None)
	d2 = pd.read_table("WikiNews_Train.tsv", header=None)
	d3 = pd.read_table("Wikipedia_Train.tsv", header=None)
	frames = [d1, d2, d3]
	table = pd.concat(frames)
	
	del d1
	del d2
	del d3
	
	for i, j in zip(table[4], table[9]):
		if i in pairs:
			pairs[i].append(j)
		else:
			pairs[i] = [j]

	for i in pairs.keys():
		if 1 in pairs[i]:
			pairs[i] = 1
		else:
			pairs[i] = 0
			
def validation_accuracy():
	global pairs
	d1 = pd.read_table("News_Dev.tsv", header=None)
	d2 = pd.read_table("WikiNews_Dev.tsv", header=None)
	d3 = pd.read_table("Wikipedia_Dev.tsv", header=None)
	frames = [d1, d2, d3]
	table = pd.concat(frames)
	
	coun = 0
	coun2 = 0
	acc = 0
	for i, j in zip(table[4], table[9]):
		if i in pairs:
			if pairs[i] == j:
				acc += 1
			coun += 1
		elif len(i) >= 10:
			if j == 1:
				acc += 1
			coun += 1
		coun2 += 1
	print("accurate predictions: ", acc, " able to predict: ", coun)
	print("total in val: ", coun2)
	print("accuracies: ", acc/coun, acc/coun2)

#baseline_with_no_length_part_training()
#baseline_training()
restrictive_training()
validation_accuracy()
