from __future__ import division
from scipy.stats import norm
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

precision_variable = 0.00001;

def setRandomSeed(seed_value):
	np.random.seed(seed_value);

def drawAgentPrice():
	price = sigma * np.random.randn() + mu;

	while (price < lower) or (price > upper):
		price = sigma * np.random.randn() + mu;     

	return price;

def drawAgentQuality(price, t_test, m0):
	n = t_test / 2; # Trial number
	t_train = t_test;
	alpha = 2.5; # Problem difficulty factor
	beta = -0.004;

	m = m0 * (1 - math.exp(beta * t_train));
	p = m * (1 - math.exp(- alpha * price));
	s = np.random.binomial(n, p);

	quality = 0.5 + (1 / n) * s;
	
	return quality*10;

def computePDF(x):
	pdf = np.exp(-np.power(x-mu, 2)/(2*np.power(sigma, 2)))/(sigma*np.sqrt(2*np.pi));

	return pdf;

def normalisePDF(pdf):
	cdf_upper = norm.cdf(upper, mu, sigma);
	cdf_lower = norm.cdf(lower, mu, sigma);

	normalisation_factor = cdf_upper - cdf_lower;

	if(normalisation_factor !=0):
		normalised_pdf = pdf/normalisation_factor;
	else:
		normalised_pdf = pdf;

	return normalised_pdf;

def computeCDF(x):
	cdf = norm.cdf(x, mu, sigma);
	
	return cdf;

def normaliseCDF(cdf):
	cdf_upper = norm.cdf(upper, mu, sigma);
	cdf_lower = norm.cdf(lower, mu, sigma);
	normalisation_factor = cdf_upper - cdf_lower;

	if(normalisation_factor !=0):
		normalised_cdf = (cdf - cdf_lower)/normalisation_factor;
	else:
		normalised_cdf = cdf;

	return normalised_cdf;

def locateBoundaries(x, y):
	i, b_index = 0, 0;
	low_boundary_set = False;
	high_boundary_set = False;
	boundaries = [];
	lower_temp = lower;

	while(lower_temp < (upper - precision_variable)):
		if(i < (len(y)-1) and low_boundary_set != True and y[i+1] > y[i]):
				low_boundary = i;
				low_boundary_set = True;
				print "lower boundary", low_boundary, x[low_boundary];

		elif(low_boundary_set == True and y[i] < y[low_boundary]):
				high_boundary = i;
				high_boundary_set = True;
				print "high_boundary", high_boundary, x[high_boundary];
				if(low_boundary_set == True and high_boundary_set == True):
					boundaries.append([]);
					boundaries[b_index].append(low_boundary);
					boundaries[b_index].append(high_boundary);
					print boundaries;
					low_boundary_set = False;
					high_boundary_set = False;
					b_index += 1;

		lower_temp += precision_variable;
		i += 1;

	if(low_boundary_set == True and high_boundary_set != True):
		high_boundary = len(x) - 1;
		boundaries.append([]);
		boundaries[b_index].append(low_boundary);
		boundaries[b_index].append(high_boundary);
		b_index += 1;

	return boundaries;

def setBaseValuation(x):

	pdf = computePDF(x);
	linePDF, = plt.plot(x, pdf,'k', linewidth=2.0, label="PDF");

	normalised_pdf = normalisePDF(pdf);
	lineNormalisedPDF, = plt.plot(x, normalised_pdf, 'r', label="normalised PDF");
	
	cdf = computeCDF(x);
	lineCDF, = plt.plot(x, cdf, 'b', linewidth=2.0, label="CDF");

	normalised_cdf = normaliseCDF(cdf);
	lineNormalisedCDF, = plt.plot(x, normalised_cdf, 'c', label="normalised CDF");

	unironed = - x - (normalised_cdf/normalised_pdf);
	line, = plt.plot(x, unironed, 'y', linewidth=2.0, label="unironed");

	return unironed;

def main():
	global lower, upper, mu, sigma;

	N = int(sys.argv[1]); 		# number of competing agents
	seed = int(sys.argv[2]);	
	lower = float(sys.argv[3]); # lower boundary of truncated normal distribution
	upper = float(sys.argv[4]); # upper boundary of truncated normal distribution
	mu = float(sys.argv[5]); 	# mean of truncated normal distribution
	sigma = float(sys.argv[6]); # variance of truncated normal distribution
	competition_number = int(sys.argv[7]); 

	setRandomSeed(seed);

	x_base = np.arange(lower, upper, precision_variable);
	unironed_baseValuation = setBaseValuation(x_base);
	boundaries_to_iron = locateBoundaries(x_base, unironed_baseValuation);

	t_test = 500;
	m0 = np.random.uniform(0,1);
	columns = 5;
		
	competition_list = [];

	for i in range(0, competition_number):
		competition_temp = np.full([N,columns, ],0);

		for j in range(0,N):
			price = drawAgentPrice();
			quality = drawAgentQuality(price, t_test, m0);
			competition_temp [j][0] = j;
			competition_temp [j][1] = price;
			competition_temp [j][2] = quality;

		competition_list.append(competition_temp);

	f = file('agentList.npy', 'w');
	np.save(f, competition_list);

	f1 = file('unironedVB.npy', 'w');
	np.save(f1, unironed_baseValuation);

	f2 = file('boundaries.npy', 'w');
	np.save(f2, boundaries_to_iron);

main();