"""
This code finds the optimal value of k (the # of nearest neighbors) for the adaptive bandwidth KDE.
"""

from location_project import kde_2d
import numpy as np
import csv

# conversitions for SoCal
KM_TO_LON = 0.010615  # = (degree longitude)/km
KM_TO_LAT = 0.008989  # = (degree latitude)/km

# load the data
path = "./data/"

train_raw = np.genfromtxt(path + "training.csv", delimiter=',')
val_raw = np.genfromtxt(path + "validation.csv", delimiter=',')
test_raw = np.genfromtxt(path + "test_3plus.csv", delimiter=',')

cols = [0, 2, 3] # user_id, lon, lat
train, val, test = train_raw[:, cols], val_raw[:, cols], test_raw[:, cols]

# learn population/indiv bandwidths & corresponding densities for various values of k
k = np.arange(1,50,2) # number of nearest neighbors to try
uid = np.unique(train[:,0]) # get the unique user_ids

bw_train_pop, bw_train_indv = np.ones((train.shape[0], k.shape[0])), np.ones((train.shape[0], k.shape[0])) 

density_train_pop, density_train_indv = np.ones((train.shape[0], k.shape[0])), np.ones((train.shape[0], k.shape[0]))
density_val_pop, density_val_indv = np.ones((val.shape[0], k.shape[0])), np.ones((val.shape[0], k.shape[0]))
density_test_pop, density_test_indv = np.ones((test.shape[0], k.shape[0])), np.ones((test.shape[0], k.shape[0]))

c = 0
for nn in k:
	# population bw/log pdf
	bw_train_pop[:, c] = kde_2d._learn_nearest_neighbors_bandwidth(train[:, 1:3], nn, KM_TO_LON, KM_TO_LAT) # calc population bw
	print "Done calculating population bw for k = %s" % nn
	
	pop = np.append(train, np.reshape(bw_train_pop[:, c], (len(train), 1)), 1)
	pop = np.append(pop, np.ones((len(train), 1)), 1)
	kde_pop = kde_2d.KDE(pop) # population KDE

	for i in range(train.shape[0]):
        	density_train_pop[i,c] = kde_pop.log_pdf(train[i, 1:3])
	print "\tDone calculating training population log pdf"
	
	for i in range(val.shape[0]):
		density_val_pop[i,c] = kde_pop.log_pdf(val[i, 1:3])
	print "\tDone calculating validation population log pdf"
	
	for i in range(test.shape[0]):
		density_test_pop[i,c] = kde_pop.log_pdf(test[i, 1:3])
	print "\tDone calculating test population log pdf"

	# individual bw/log pdf
	for user in range(uid.shape[0]):
		train_index, val_index, test_index = np.where(train[:, 0] == uid[user]), np.where(val[:, 0] == uid[user]), np.where(test[:, 0] == uid[user])
		train_user = train[train_index[0], :]

		bw_train_indv[train_index[0], c] = kde_2d._learn_nearest_neighbors_bandwidth(train_user, nn, KM_TO_LON, KM_TO_LAT) # calc indv bw
		indv = np.append(train_user, np.reshape(bw_train_indv[train_index[0], c], (len(train_user), 1)), 1)
		indv = np.append(indv, np.ones((len(train_user), 1)), 1)
		indv_kde = kde_2d.KDE(indv) # individual KDE

		for i in train_index[0]:
			density_train_indv[i, c] = indv_kde.log_pdf(train[i, 1:3])

		for i in val_index[0]:
			density_val_indv[i, c] = indv_kde.log_pdf(val[i, 1:3])

		# if user has data in test set, calculate the log pdf for his/her tweets
		if test_index[0].shape[0] > 0:
			for i in test_index[0]:
				density_test_indv[i, c] = indv_kde.log_pdf(test[i, 1:3])

	c += 1

# save results to file
path = path + "parameter_tuning/" 
names = ["bw_train_pop", "bw_train_indv", "density_train_pop", "density_train_indv", \
        "density_val_pop", "density_val_indv", "density_test_pop", "density_test_indv"]

for out in names:
	with open(path + out + ".csv", "wb") as f:
		w = csv.writer(f)
		w.writerows(eval(out))








