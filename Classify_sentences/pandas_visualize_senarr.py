# Created by Jamilah Foucher, 21/02/2022

# Visualizing a sentence array in pandas

# Example of how sen and senarray should look (use make_a_properlist_str to get sentence array in this form)
# senarray = [['How', 'can', 'I', 'be', 'a', 'good', 'geologist?'],
# ['What', 'should', 'I', 'do', 'to', 'be', 'a', 'great', 'geologist?']]


import pandas as pd


def pandas_visualize_senarr(senarray):

	grp4pandas = []
	for i in range(len(senarray)):
		arr = []
		for j in range(len(senarray[i])):
			arr.append([senarray[i][j]])
		grp4pandas.append(arr)

	df_grp = pd.DataFrame(data=grp4pandas)
	
	return df_grp