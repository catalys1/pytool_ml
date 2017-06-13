#!/usr/bin/python

import argparse
import json
from tools import Matrix
from cluster import KMeansCluster


class MLManager(object):

	def go(self):
		'''
		'''
		args = self.parse()
		filepath = args.A
		k = args.K
		col_del = args.D
		norm = args.N
		parameterize = args.P

		data_matrix = Matrix(filepath)
		for c in col_del:
			data_matrix.split_class(int(c))

		if norm:
			data_matrix.normalize()

		print 'Dataset name: {}'.format(data_matrix.dataset_name)
		print 'Number of instances: {}'.format(data_matrix.rows())
		print 'Number of attributes: {}'.format(data_matrix.cols())

		if not parameterize:
			self.runKMeans(k,data_matrix)
		else:
			self.runParameterization(data_matrix)


	def runKMeans(self, k, data_matrix):
		print 'K-Means with k = {}\n'.format(k)

		learner = KMeansCluster()
		results = learner.cluster(data_matrix, k)
		print 'Number of clusters: {}'.format(results[0])
		print 'Centroids:\n{}'.format(learner.printCentroids())
		print 'Instances in each cluster:\n  {}'.format(
			'\n  '.join('Cluster {}: {}'.format(int(i),x) for i,x in sorted(results[2].items())))
		print 'Cluster SSE:\n  {}'.format(
			'\n  '.join('Cluster {}: {:.3f}'.format(i,x) for i,x in enumerate(results[3])))
		print 'Total SSE: {:.3f}'.format(results[4])

		score = learner.silhouette()
		print 'Silhouette score: {:.3f}'.format(score)

		sses = json.load(open('error.json'))
		sses[str(k)] = results[4]
		json.dump(sses, open('error.json', 'w'))

		scores = json.load(open('scores.json'))
		scores[str(k)] = score
		json.dump(scores, open('scores.json', 'w'))



	def runParameterization(self, data_matrix):
		print 'K-Means parameterization'

		learner = KMeansCluster()
		results = learner.parameterize(data_matrix)

		print 'Best number of clusters: {}'.format(results[0])
		print 'Mean Centroids:\n{}'.format(learner.printCentroids())
		print 'Instances in each cluster:\n  {}'.format(
			'\n  '.join('Cluster {}: {}'.format(int(i),x) for i,x in sorted(results[2].items())))
		print 'Cluster SSE:\n  {}'.format(
			'\n  '.join('Cluster {}: {:.3f}'.format(i,x) for i,x in enumerate(results[3])))
		print 'Total SSE: {:.3f}'.format(results[4])



	def parse(self):

		parser = argparse.ArgumentParser()
		
		parser.add_argument('-K', metavar='#CLUSTERS', type=int, default=4,
			help='Number of clusters for K-Means')
		parser.add_argument('-A', metavar='ARFF file path', type=str, required=True,
			help='Path to ARFF file')
		parser.add_argument('-D', metavar='col', nargs='*', default=[],
			help='Indices of columns to remove from the dataset')
		parser.add_argument('-N', action='store_true', default=False,
			help='Normalize continous attributes')
		parser.add_argument('-P', action='store_true', default=False,
			help='Parameterize: find the best k, and a good clustering for that k')

		args = parser.parse_args()
		return args


if __name__ == '__main__':
	MLManager().go()