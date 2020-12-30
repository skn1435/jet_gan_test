import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_image(data, n_image = 16, n_x = 4, n_y = 4) :
	# figure
	fig = plt.figure()
	# divide
	a_c = np.array([])
	i_image = 0
	for i_cx in range(n_x) :
		for i_cy in range(n_y) :
			i_image += 1
			a_c = np.append(a_c, fig.add_subplot(n_x, n_y, i_image))
	# plot
	for i_c in range(n_image) :
		(a_c[i_c]).imshow(data[i_c].reshape([28, 28]), cmap = "gray")
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
	plot_image(data)


if __name__ == "__main__" :
	main()
