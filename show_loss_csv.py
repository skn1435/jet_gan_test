import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(data, label) :
	n_data = len(data)
	n_x = len(data[0])
	a_x = np.arange(n_x)
	for i_data in range(n_data) :
		plt.plot(a_x, data[i_data], label = label[i_data])
	plt.yscale("log")
	plt.legend()
	plt.show()

def main() :
	# args
	n_args = len(sys.argv)
	if n_args < 2 :
		print("usage: python3 %s <filename>" % (sys.argv[0]))
		exit()
	# load data
	fname = sys.argv[1]
	df = pd.read_csv(fname, index_col = 0)
	print(df)
	data = df.values
	# plot
	plot_loss(data, ["D", "G"])


if __name__ == "__main__" :
	main()
