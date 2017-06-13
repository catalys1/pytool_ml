import numpy as np
from collections import Counter

class KMeansCluster(object):
	
	def __init__(self):
		self.feature_matrix = None
		self.centroids = None
		self.clusters = None
		self.k = None


	def cluster(self, feature_matrix, k=4):
		'''
		'''
		self.feature_matrix = feature_matrix
		self.k = k
		con = np.array(feature_matrix.continuous_cols(), np.uint32)
		nom = np.array(feature_matrix.nominal_cols(), np.uint32)

		data = feature_matrix.data

		# centroids = np.array([feature_matrix[i,:].copy() for i in xrange(k)])
		centroids = data[np.random.choice(data.shape[0],k),:]
		clusters = np.zeros(feature_matrix.rows())

		error = None
		SSE_d1 = -1
		SSE = 0
		n = 1

		while (np.around(SSE_d1) != np.around(SSE)):
			# print '***************'
			# print 'Iteration {}'.format(n)
			# print '***************'
			# recalculate the centroids
			if SSE_d1 != -1:
				self._compute_centroids(data, centroids, clusters, con, nom)
			# print 'Computing Centroids:'
			# for i,c in enumerate(centroids):
			# 	print 'Centroid {} = {}'.format(i, self._print_centroid(c,feature_matrix))
			# figure out which cluster each instance belongs to
			# print 'Making Assignments'
			errors = self._assign_instances(data, centroids, clusters, con, nom)

			SSE_d1 = SSE
			SSE = np.sum(errors)
			# print '    {}'.format('\n    '.join(self._print_clusters(clusters)))
			# print 'SSE: {:.3f}\n'.format(SSE)

			n+=1

		self.centroids = centroids
		self.clusters = clusters
		cluster_sizes = Counter(clusters)
		# Report:
		# a) The number of clusters (k)
		# b) The centroid values for each cluster (centroids)
		# c) The number of instances for each centroid (cluster_sizes)
		# d) SSE of each cluster (errors)
		# e) Total SSE (SSE)
		return (k, centroids, cluster_sizes, errors, SSE)


	def silhouette(self):
		'''Can be called only after clustering
		'''
		# calculate a(i)
		features = self.feature_matrix
		clusters = self.clusters
		k = self.k
		con = np.array(features.continuous_cols(), np.uint32)
		nom = np.array(features.nominal_cols(), np.uint32)

		sils = np.zeros(k)
		inds = np.arange(features.rows())
		for i in inds:
			inst = features[i,:]
			data = features[inds!=i,:]
			clust = clusters[i]

			con_dis = np.power(inst[con] - data[:,con], 2)
			con_dis[~np.isfinite(con_dis)] = 1
			nom_dis = np.where(inst[nom]==data[:,nom], 0, 1)
			nom_dis[:,np.isinf(inst[nom])] = 1
			sq_dist = np.sum(con_dis,axis=1) + np.sum(nom_dis, axis=1)

			clu_less_1 = clusters[inds!=i]
			bs = []
			for j in set(range(k))-set([clust]):
				bs.append(np.mean(sq_dist[clu_less_1==j]))

			b_i = np.min(bs)
			a_i = np.mean(sq_dist[clu_less_1==clust])
			s_i = (b_i - a_i)/np.max([b_i,a_i])

			sils[int(clust)] += s_i

		sils = np.divide(sils, [np.count_nonzero(clusters==i) for i in xrange(k)])
		total = np.mean(sils)
		return total


	def parameterize(self, feature_matrix, kmin=2, kmax=10, iters=5):
		'''
		'''
		scores = dict()
		for k in xrange(kmin,kmax+1):
			try:
				self.cluster(feature_matrix, k)
				score = self.silhouette()
				scores[k] = score
			except IndexError, e:
				print e
				print k
				raise e

		best_k = sorted(scores.items(), key=lambda x:x[1], reverse=True)[0][0]

		centroids = []
		for i in xrange(iters):
			results = self.cluster(feature_matrix, best_k)
			centroids.append(results[1])

		# Cluster the centroids
		con = np.array(feature_matrix.continuous_cols(), np.uint32)
		nom = np.array(feature_matrix.nominal_cols(), np.uint32)
		data = np.concatenate(centroids, axis=0)
		mean_cents = data[:best_k,:].copy()
		clusters = np.zeros(len(data))
		self._cluster(data, mean_cents, clusters, con, nom)

		data = feature_matrix.data
		clusters = np.zeros(feature_matrix.rows())
		errs = self._assign_instances(feature_matrix.data, mean_cents,clusters,con,nom)

		SSE = np.sum(errs)
		cluster_counts = Counter(clusters)

		return (best_k, mean_cents, cluster_counts, errs, SSE)


	def printCentroids(self):
		'''
		'''
		cents = []
		for i,c in enumerate(self.centroids):
			cents.append('  Centroid {} = {}'.format(i, self._print_centroid(c,self.feature_matrix)))
		return '\n'.join(cents)


	def _cluster(self, data, centroids, clusters, con, nom):

		error = None
		SSE_d1 = -1
		SSE = 0

		while (np.around(SSE_d1) != np.around(SSE)):
			if SSE_d1 != -1:
				self._compute_centroids(data, centroids, clusters, con, nom)
			errors = self._assign_instances(data, centroids, clusters, con, nom)
			SSE_d1 = SSE
			SSE = np.sum(errors)


	def _compute_centroids(self, data, centroids, clusters, con, nom):
		'''
		'''		
		for i in xrange(centroids.shape[0]):
			# Ignore 'unknown' values (encoded as infinity), unless all the
			# instances in a cluster have 'unknown' for an attribute - in that
			# case make 'unknown' the centroid value
			clust_inds = clusters==i
			clust_data = data[clust_inds]
			con_clust = clust_data[:,con]
			con_clust = np.ma.masked_array(con_clust, np.isinf(con_clust))
			centroids[i,con] = np.mean(con_clust, axis=0).data
			# calculate centroid values for nominals
			nom_clust = clust_data[:,nom]
			centroids[i,nom] = [
				sorted(
					Counter(nom_clust[np.isfinite(nom_clust[:,j]),j].flatten()
					).most_common(),
					key=lambda x:x[::-1],
					reverse=True
				)[0][0]
				for j in xrange(len(nom))
			]
			nom_clust = np.ma.masked_array(nom_clust, np.isinf(nom_clust))
			# Take care of the cases where all the instances have a missing
			# value for the same attribute
			if np.any(con_clust.mask):
				centroids[i,con[np.all(con_clust.mask, axis=0)]] = np.infty
			if np.any(nom_clust.mask):
				centroids[i,nom[np.all(nom_clust.mask, axis=0)]] = np.infty


	def _assign_instances(self, data, centroids, clusters, con, nom):
		'''
		'''
		errors = np.zeros(centroids.shape[0])
		for i in xrange(data.shape[0]):
			inst = data[i,:]
			con_dis = np.power(inst[con] - centroids[:,con], 2)
			con_dis[~np.isfinite(con_dis)] = 1
			nom_dis = np.where(inst[nom]==centroids[:,nom], 0, 1)
			nom_dis[:,np.isinf(inst[nom])] = 1
			sq_dist = np.sum(con_dis,axis=1) + np.sum(nom_dis, axis=1)
			cluster_index = np.argmin(sq_dist) 
			clusters[i] = cluster_index
			errors[cluster_index] += sq_dist[cluster_index]
		return errors


	def _print_centroid(self, centroid, mat):
		'''
		'''
		return ', '.join(
			'?' 
			if centroid[i]==np.infty 
			else (
				"'{}'".format(mat.nominal_enum[i][centroid[i]])
				if i in mat.nominal_enum 
				else '{:.3f}'.format(centroid[i])
				)
			for i in xrange(len(centroid))
		)


	def _print_clusters(self, clusters):
		'''
		'''
		lines = []
		for i in xrange(0, len(clusters), 10):
			line = ['{}={}'.format(j,int(clusters[j])) for j in xrange(i,min(i+10, len(clusters)))]
			lines.append(line)
		lines = [' '.join(x) for x in lines]
		return lines
				