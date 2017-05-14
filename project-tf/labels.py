#!/usr/bin/env python

def main():
	f = open('stage1_solution.csv')
	ones = 0
	zeros = 0
	total = 0
	for line in f:
		if line[:3] == 'id,':
			continue
		line = line.strip().split(',')
		label = int(line[1])
		if label == 1:
			ones += 1
		total += 1
	zeros = total-ones
	print float(zeros)/total
	f.close()

if __name__ == '__main__':
	main()