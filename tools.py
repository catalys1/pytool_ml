import re
import numpy as np
from collections import defaultdict
import random
# from matrix import matrix


def mode(a, axis=0):
	# taken from scipy code
	# https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py#L609
	scores = np.unique(np.ravel(a))       # get ALL unique values
	testshape = list(a.shape)
	testshape[axis] = 1
	oldmostfreq = np.zeros(testshape)
	oldcounts = np.zeros(testshape)

	for score in scores:
		template = (a == score)
		counts = np.expand_dims(np.sum(template, axis),axis)
		mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
		oldcounts = np.maximum(counts, oldcounts)
		oldmostfreq = mostfrequent

	return mostfrequent, oldcounts


def parseArff(filepath):
	'''Parses an arff file.

	Parameters
	----------
	filepath : string
		Path to the arff file

	Returns
	-------
	A 3-tuple containing
	(name-of-dataset, list-of-attributes, list-of-data-instances).
	'''
	# This function parses the arff file using regular expressions.
	# Open the arff file and read it in as one big string
	s = None
	with open(filepath, 'r') as f:
		s = f.readlines()

	# Remove commented lines
	s = '\n'.join(x for x in s if not x.startswith('%')).strip()

	# Use a pattern to extract the name of the relation
	rel_p = '@RELATION[\s]*([^\s]*)'
	relation_name = re.search(rel_p, s, flags=re.I).group(1)

	# Use a pattern to extract the name and possible values for each attribute
	att_p = '@ATTRIBUTE[\s]*([^\s]*)[\s]*([^\s]*)'
	attributes = re.findall(att_p, s, flags=re.I)
	# Post processing on attributes to get them into a more useful form:
	#   (attr-name, 'continuous') or (attr-name, [nom-val-1,...,nom-val-n])
	# enum is a dictionary from the position of nominal-value columns to
	#   dictionaries of {value:position} for nominal values
	enum = {}
	for i,a in enumerate(attributes):
		if a[1].startswith('{'):  # nominal valued attribute
			attributes[i] = (a[0],re.split('[\s]*,[\s]*',a[1][1:-1]))
			enum[i] = {x:float(j) for j,x in enumerate(attributes[i][1])}
		else:  # continuous valued attribute
			attributes[i] = (a[0], a[1].lower())

	# Construct a pattern to match data instances based on the type and position
	# of each attribute
	cont_attr_p = '(-?[\d\.]*|\?)'  # match pos/neg numbers with or without decimals
	inst_p = ',[\s]*'.join([
		cont_attr_p if type(x[1]) == str
		else '(\?|{})'.format('|'.join(['(?:{})'.format(a) for a in x[1]]))
		for x in attributes])
	inst_p = inst_p.replace('+', '\+')

	# A little function to convert values from an instance into floats. If the
	# value is continuous, we cast it to a float, otherwise we look up it's
	# position in the enum
	def convert_nums(n, i):
		'''n: value
		i: position of value in instance
		'''
		try:
			n = float(n)
		except:
			if n != '?':
				n = enum[i][n]
			else:
				n = np.infty
		return n

	# Find all the instances and put them in a suitable form. By making all
	# the values floats, we can easily turn the instance array into a numpy
	# array
	# Look at everything below the @DATA marker
	instances = re.split('@DATA',s,flags=re.I)[1]
	instances = [
		[convert_nums(a,i) for i,a in enumerate(x.groups())] 
		for x in re.finditer(inst_p, instances)]

	return (relation_name, attributes, instances)



class Matrix(object):
	'''The Matrix class is a container for the data and metadata of a dataset.
	It also contains some useful functions for manipulating and working with
	the data.
	'''

	# Missing values in the dataset will be replaced with np.infty
	MISSING = np.infty

	def __init__(self, arff_file=None):
		'''Initialize a matrix.
		If arff_file is none, then create an empty matrix. Otherwise populate
		the matrix based on the data in the arff_file. arff_file is a string
		with the path to the file.
		'''
		self.dataset_name = 'Untitled'
		self.data = []
		self.attributes = []
		self.nominal_enum = dict()

		if arff_file is not None:
			self.load_arff(arff_file)


	def __getitem__(self, key):
		'''x.__getitem__(y) <==> x[y]'''
		return self.data[key]


	def __setitem__(self, key, value):
		'''x.__setitem__(y,v) <==> x[y] = v'''
		return self.data[key]


	def __str__(self):
		return '<Dataset: {} - {} instances, {} attributes>'.format(
			self.dataset_name, *self.data.shape)


	def __repr__(self):
		return self.__str__()


	def split_class(self, class_col=-1):
		'''m2 = m1.split_class(class_col=-1)
		Split the matrix into feature and label matrices. m1 retains the
		feature columns and m2 retains the class column. The information for 
		the label matrix is taken from m1, and also removed from m1. This 
		function assumes that there is only one class-label column. The default
		label column is the last column.
		'''
		# Convert a negative index to a positive index
		att_ind = class_col % len(self.attributes)
		# Create a new matrix and add the data/metadata for the specified col
		# Also remove the data/metadata from *self*
		label_mat = Matrix()
		label_mat.dataset_name = self.dataset_name
		attr_name = self.attributes[att_ind]
		self.attributes.remove(attr_name)
		label_mat.attributes.append(attr_name)
		if att_ind in self.nominal_enum:
			val = self.nominal_enum.pop(att_ind)
			label_mat.nominal_enum = dict(
				[[att_ind, val]])
		new_enum = dict()
		for k, d in self.nominal_enum.items():
			if k > att_ind:
				new_enum[k-1] = d
			else:
				new_enum[k] = d
		self.nominal_enum = new_enum
		label_mat.data = self.data[:,class_col].reshape((self.data.shape[0],1))
		inds = np.arange(self.data.shape[1]) != att_ind
		self.data = self.data[:,inds]
		return label_mat


	def split(self, args):
		'''TODO: implement
		'''
		pass


	def load_arff(self, arff_file):
		'''Populate the matrix with the data from an arff file'''
		relation_name, attributes, instances = parseArff(arff_file)
		self.dataset_name = relation_name
		for i, a in enumerate(attributes):
			self.attributes.append(a[0])
			if not type(a[1]) is str:
				self.nominal_enum[i] = {x:j for j,x in enumerate(a[1])}
				self.nominal_enum[i].update({j:x for j,x in enumerate(a[1])})
		self.data = np.array(instances)


	def normalize(self):
		'''Normalize each continuous-valued column'''
		for i in xrange(self.cols()):
			if not i in self.nominal_enum:
				a = np.ma.masked_equal(self.data[:,i], self.MISSING).compressed()
				min_val = np.min(a)
				max_val = np.max(a)
				rdiff = max_val-min_val
				self.data[:,i] = np.divide(
					np.subtract(self.data[:,i], min_val), rdiff)


	def shuffle(self, buddy=None):
		'''Shuffle the row order. If a buddy Matrix is provided, it will be 
		shuffled in the same order. *buddy* must have the same number of rows
		in its data as this Matrix.
		'''
		if buddy is not None:
			np.random.shuffle(self.data)
		else:
			inds = np.random.shuffle(np.arange(self.data.shape[0]))
			self.data = self.data[inds]
			buddy.data = buddy.data[inds]


	def column_mean(self, col):
		"""Get the mean of the specified column"""
		a = np.ma.masked_equal(self.col(col), self.MISSING).compressed()
		return np.mean(a)


	def rows(self):
		'''Return the number of rows'''
		return len(self.data)


	def cols(self):
		'''Return the number of columns'''
		return len(self.attributes)


	def continuous_cols(self):
		'''Return a list of the column indices that correspond to continuous
		attributes.
		'''
		return list(set(range(self.cols())) - set(self.nominal_enum.keys()))


	def nominal_cols(self):
		'''Return a list of the column indices that correspond to nominal
		attributes.
		'''
		return self.nominal_enum.keys()


	def attr_name(self, col):
		"""Get the name of the specified attribute"""
		return self.attributes[col]


	def attr_value(self, attr, val):
		'''Get the name of the specified value (attr is a column index)
		:param attr: index of the column
		:param val: index of the value in the column attribute list
		:return:
		'''
		return self.nominal_enum[attr][int(val)]


	def value_count(self, col):
		"""
		Get the number of values associated with the specified attribute (or columnn)
		0=continuous, 2=binary, 3=trinary, etc.
		"""
		if col in self.nominal_enum:
			return len(self.nominal_enum[col]) / 2
		else:
			return 0


	def column_min(self, col):
		"""Get the min value in the specified column"""
		a = np.ma.masked_equal(self.data[:,col], self.MISSING).compressed()
		return np.min(a)


	def column_max(self, col):
		"""Get the max value in the specified column"""
		a = np.ma.masked_equal(self.data[:,col], self.MISSING).compressed()
		return np.max(a)


	def most_common_value(self, col):
		"""Get the most common value in the specified column"""
		a = np.ma.masked_equal(self.data[:,col], self.MISSING).compressed()
		(val, count) = mode(a)
		return val[0]
