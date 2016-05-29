from __future__ import division
from scipy.stats import norm
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

precision_variable = 0.00001;


def drawAgentPrice(lower, upper, mu, sigma):
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

def computePDF(mu, sigma, x):
	pdf = np.exp(-np.power(x-mu, 2)/(2*np.power(sigma, 2)))/(sigma*np.sqrt(2*np.pi));

	return pdf;

def normalisePDF(mu, sigma, upper, lower, pdf):
	cdf_upper = norm.cdf(upper, mu, sigma);
	cdf_lower = norm.cdf(lower, mu, sigma);

	normalisation_factor = cdf_upper - cdf_lower;

	if(normalisation_factor !=0):
		normalised_pdf = pdf/normalisation_factor;
	else:
		normalised_pdf = pdf;

	return normalised_pdf;

def computeCDF(mu, sigma, x):
	cdf = norm.cdf(x, mu, sigma);
	
	return cdf;

def normaliseCDF(mu, sigma, upper, lower, cdf):
	cdf_upper = norm.cdf(upper, mu, sigma);
	cdf_lower = norm.cdf(lower, mu, sigma);
	normalisation_factor = cdf_upper - cdf_lower;

	if(normalisation_factor !=0):
		normalised_cdf = (cdf - cdf_lower)/normalisation_factor;
	else:
		normalised_cdf = cdf;

	return normalised_cdf;

def computeVirtualValuation(price, quality):
	ironed_value = findIronedValue(price);
	virtual_valuation = quality + ironed_value;

	return virtual_valuation;

def computeActualValuation(price, quality):	
	actual_valuation = quality - price;

	return actual_valuation;

def findAgentWithValuation(data, index, position):
	agent = sorted(data, key=lambda entry: entry[4], reverse=True); #sorting is expensive
	
	return agent[position][:];

def ironVirtuality(data, X):
	virtuality = [];
	agent = sorted(data, key=lambda entry: entry[1]); 
	
	#print agent;
	for i in range (0, len(data)-1):
		if(agent[i+1][4] > agent[i][4]):
			agent[i+1][4] = agent[i][4];

	#print agent;
	data = sorted(agent, key=lambda entry: entry[0]); 
	
	for i in range (0, len(X)):
		virtuality.insert(i, data[i][4]);
	
	#print "\nvirtuality\n", virtuality;
	plt.scatter(X, virtuality, color ="r");

	return virtuality;

def locateBoundaries(lower, upper, x, y):
	i, b_index = 0, 0;
	low_boundary_set = False;
	high_boundary_set = False;
	boundaries = [];

	while(lower < (upper - precision_variable)):
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
					#print boundaries;
					low_boundary_set = False;
					high_boundary_set = False;
					b_index += 1;

		lower += precision_variable;
		i += 1;

	if(low_boundary_set == True and high_boundary_set != True):
		high_boundary = len(x) - 1;
		boundaries.append([]);
		boundaries[b_index].append(low_boundary);
		boundaries[b_index].append(high_boundary);
		b_index += 1;

	return boundaries;

def setBaseValuation(lower, upper, mu, sigma, x):

	pdf = computePDF(mu, sigma, x);
	linePDF, = plt.plot(x, pdf,'k', linewidth=2.0, label="PDF");

	normalised_pdf = normalisePDF(mu, sigma, upper, lower, pdf);
	lineNormalisedPDF, = plt.plot(x, normalised_pdf, 'r', label="normalised PDF");
	
	cdf = computeCDF(mu, sigma, x);
	lineCDF, = plt.plot(x, cdf, 'b', linewidth=2.0, label="CDF");

	normalised_cdf = normaliseCDF(mu, sigma, upper, lower, cdf);
	lineNormalisedCDF, = plt.plot(x, normalised_cdf, 'c', label="normalised CDF");

	unironed = - x - (normalised_cdf/normalised_pdf);
	line, = plt.plot(x, unironed, 'y', linewidth=2.0, label="unironed");

	#boundaries = locateBoundaries(lower, upper, x, unironed);
	#print "boundaries to iron:", boundaries;

	#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=3, mode="expand", borderaxespad=0.);
	#plt.savefig('test.png');
	#plt.show();

	return unironed;

def findIronedValue(price):
	count = 0;
	value_set = False;
	x_index = int(price/precision_variable);
	#print "x_index", x_index;
	#print "corresponding y", y[x_index];

	for j in range(0, len(boundaries_to_iron)):
		if(x_index >= boundaries_to_iron[j][0] and x_index <= boundaries_to_iron[j][1]):
			y_ironed = unironed_baseValuation[boundaries_to_iron[j][0]];
			value_set = True;

	if(value_set == False):
		y_ironed = unironed_baseValuation[x_index];

	#print "ironed y", y_ironed;
	return y_ironed;

def computeOptimalReverseAuction(lower, upper, mu, sigma, data, N):
	print "\n\nOptimal Reverse Auction\n\n"

	for i in range (0,N):
		data[i][3] = computeVirtualValuation(data[i][1], data[i][2]);

	#determine optimal value of reserve parameters
	price_reserve = 3.5;
	quality_reserve = 10;
	valuation_reserve = computeVirtualValuation(price_reserve, quality_reserve);

	information_rent, utility_of_principal, prize = 0, 0, 0; 
	valuation_b, b_index = 0, 0;
	valuation_sb, sb_index = 0, 0;
	
	for i in range (0,N):
		valuation_temp = data[i][3];
		if((valuation_temp > valuation_b) or (valuation_temp == valuation_b and data[i][1] < data[b_index][1])):
			valuation_sb = valuation_b;
			sb_index = b_index;
			valuation_b = valuation_temp;
			b_index = i;
		elif(valuation_temp > valuation_sb):
			valuation_sb = valuation_temp;
			sb_index = i;
	
	print "Best:", b_index;
	print "\twith virtual valuation", valuation_b;
	print "2nd Best:", sb_index;
	print "\twith virtual valuation", valuation_sb;
	
	if(valuation_b >= valuation_reserve):
		prize = binarySearchAlgorithm(upper,lower,valuation_sb,data[b_index][2], computeVirtualValuation);
		print "\nPrize:", prize;
		print "\tWinner:", b_index;
		information_rent = prize - data[b_index][1]; 
		utility_of_principal = data[b_index][2] - prize;
		
	else: 
		print "\tNo Winner";

	print "\tInformation Rent:", information_rent;
	print "\tUtility of Principal:", utility_of_principal;	

def computeEfficientReverseAuction(lower, upper, data, N):
	print "\n\nEfficient Reverse Auction\n\n"

	#determine optimal value of reserve parameters
	price_reserve = 3.5;
	quality_reserve = 10;
	valuation_reserve = computeActualValuation(price_reserve, quality_reserve);

	information_rent, utility_of_principal, prize = 0, 0, 0; 
	valuation_b, b_index = 0, 0;
	valuation_sb, sb_index = 0, 0;
	
	for i in range (0,N):
		data[i][4] = computeActualValuation(data[i][1], data[i][2]);
		valuation_temp = data[i][4];
		if((valuation_temp > valuation_b) or (valuation_temp == valuation_b and data[i][1] < data[b_index][1])):
			valuation_sb = valuation_b;
			sb_index = b_index;
			valuation_b = valuation_temp;
			b_index = i;
		elif(valuation_temp > valuation_sb):
			valuation_sb = valuation_temp;
			sb_index = i;
	
	print "Best:", b_index;
	print "\twith actual valuation", valuation_b;
	print "2nd Best:", sb_index;
	print "\twith actual valuation", valuation_sb;

	if(valuation_b >= valuation_reserve):
		prize = binarySearchAlgorithm(upper,lower,valuation_sb,data[b_index][2], computeActualValuation);
		print "\nPrize:", prize;
		print "\tWinner:", b_index;
		information_rent = prize - data[b_index][1]; 
		utility_of_principal = data[b_index][2] - prize;
		
	else: 
		print "\tNo Winner";

	print "\tInformation Rent:", information_rent;
	print "\tUtility of Principal:", utility_of_principal;	

def binarySearchAlgorithm(upper, lower, valuation_sb, b_quality, functionName):
	#print "\nBINARY SEARCH\n"
	error = 1;
	count = 0;

	#while (error != 0):
	while (np.absolute(error) > precision_variable):
	
		mid = (lower + upper)/2;
		valuation_bmax = functionName(mid, b_quality);

		if valuation_bmax < valuation_sb:
			upper = mid;
		elif valuation_bmax > valuation_sb:
			lower = mid;

		error = valuation_bmax - valuation_sb;
		#print "ERROR: ", error

	return mid;

def frange(start, stop, step):
	i = start;
	while (i < stop):
		yield i;
		i += step;

def setRandomSeed(seed_value):
 	np.random.seed(seed_value);
 	#print "seed set";

def findMaximizingPrize(data, utility):
	max_prize = 0;

	for i in range (0, len(data)):
		temp_prize = data[i][2] - utility;
		#print "temp_prize", temp_prize;
		#print "max_prize", max_prize;
	
		if(temp_prize > max_prize):
			max_prize = temp_prize;

	return max_prize;

def findOptimalPrize(competition_results):
	max_utility = 0;
	
	for i in range(0, len(competition_results)):
		if(competition_results[i][1] > max_utility):
			max_utility = competition_results[i][1];
			optimal_prize =  competition_results[i][0];
 
 	#print "\n\nOptimal prize is\n\n", optimal_prize;
	return optimal_prize;

def computeOptimalFixedPriceCompetitionPrize(lower, upper, competition_list):
	price_number = 10;
	step = (upper - lower)/price_number;
	competition_results = [];
	c_index = 0;

	for p in frange(lower, upper, step): 
		
		total_utility = 0;
		#print "\nPrize:", p;
	
		for i in range(0, len(competition_list)):
			winner_utility = computeFixedPriceCompetitionMultiple(competition_list[i], p);
			total_utility += winner_utility;

		average_utility = total_utility/len(competition_list);
		#print "\tAverage Utility: ", average_utility;

		competition_results.append([]);
		competition_results[c_index].append(p);
		competition_results[c_index].append(average_utility);
		c_index += 1;

	print "average utility per price", competition_results;
	optimal_prize = findOptimalPrize(competition_results);

	return optimal_prize;

def computeFixedPriceCompetitionMultiple(data, prize):
	winner_index = np.nan;
	max_quality, information_rent, utility = 0, 0, 0;

	for j in range (0, len(data)):
		if(data[j][1] <= prize):
			if(data[j][2] == max_quality):
				if(data[j][1] < data[winner_index][1]):
					winner_index = j;
					max_quality = data[j][2];
			elif(data[j][2] > max_quality):
				winner_index = j;
				max_quality = data[j][2];

	if isinstance(winner_index, int):
		information_rent = prize - data[winner_index][1];
		utility = data[winner_index][2] - prize;

	return utility;

def computeFixedPriceCompetition(data, prize):
	print "\n\nFixed Price Competition\n\n"

	winner_index = np.nan;
	max_quality, information_rent, utility = 0, 0, 0;

	print "Prize:", prize;	

	for j in range (0, len(data)):
		if(data[j][1] <= prize):
			if(data[j][2] == max_quality):
				if(data[j][1] < data[winner_index][1]):
					winner_index = j;
					max_quality = data[j][2];
			elif(data[j][2] > max_quality):
				winner_index = j;
				max_quality = data[j][2];

	if isinstance(winner_index, int):
		information_rent = prize - data[winner_index][1];
		utility = data[winner_index][2] - prize;
		print "\tWinner:", winner_index;

	else:
		print "\tNo Winner"

	print "\tInformation Rent:", information_rent;
	print "\tUtility of Principal:", utility;	

	return utility;

def computeFixedPriceCompetitionOld(lower, upper, data, N, optimal_prize):
	print "\n\nFixed Price Competition\n\n"

	#prize = [2, 2.5, 3.5, 3, 4, optimal_prize]; 
	prize = [optimal_prize];

	for i in range (0, len(prize)):
		winner_index = np.nan;
		max_quality = 0;
		information_rent = 0;
		utility = 0;

		for j in range (0, N):
			if(data[j][1] <= prize[i]):
				if(data[j][2] == max_quality):
					if(data[j][1] < data[winner_index][1]):
						winner_index = j;
						max_quality = data[j][2];
				elif(data[j][2] > max_quality):
					winner_index = j;
					max_quality = data[j][2];

		print "\nPrize:", prize[i];
		if isinstance(winner_index, int ):
			information_rent = prize[i] - data[winner_index][1];
			utility = data[winner_index][2] - prize[i];
			print "\tWinner:", winner_index;

		else:
			print "\tNo Winner"

		print "\tInformation Rent:", information_rent;
		print "\tUtility of Principal:", utility;	

def main():

	N = int(sys.argv[1]); 		# number of competing agents
	seed = int(sys.argv[2]);	
	lower = float(sys.argv[3]); # lower boundary of truncated normal distribution
	upper = float(sys.argv[4]); # upper boundary of truncated normal distribution
	mu = float(sys.argv[5]); 	# mean of truncated normal distribution
	sigma = float(sys.argv[6]); # variance of truncated normal distribution
	competition_number = int(sys.argv[7]); 

	setRandomSeed(seed);

	#N = 11; 
	#lower, upper, mu, sigma = 0, 7, 3.5, 2;

	global x_base, unironed_baseValuation, boundaries_to_iron;
	x_base = np.arange(lower, upper, precision_variable);
	unironed_baseValuation = setBaseValuation(lower, upper, mu, sigma, x_base);
	boundaries_to_iron = locateBoundaries(lower, upper, x_base, unironed_baseValuation);
	#print "boundaries to iron:", boundaries_to_iron;

	t_test = 500;
	m0 = np.random.uniform(0,1);
	columns = 5;
	
	data = np.full([N,columns, ],0);

	for i in range (0,N):
		price = drawAgentPrice(lower, upper, mu, sigma);
		quality = drawAgentQuality(price, t_test, m0);
		data [i][0] = i;
		data [i][1] = price;
		data [i][2] = quality;

	#set competition schemes
	competition_list = [];

	for i in range(0, competition_number):
		competition_temp = np.full([N,columns, ],0);

		for j in range(0,N):
			price = drawAgentPrice(lower, upper, mu, sigma);
			quality = drawAgentQuality(price, t_test, m0);
			competition_temp [j][0] = j;
			competition_temp [j][1] = price;
			competition_temp [j][2] = quality;

		competition_list.append(competition_temp);

	optimal_prize = computeOptimalFixedPriceCompetitionPrize(lower, upper, competition_list);

	
	computeOptimalReverseAuction(lower, upper, mu, sigma, data, N);
	computeEfficientReverseAuction(lower, upper, data, N);
	computeFixedPriceCompetition(data, optimal_prize);
	#computeFixedPriceCompetitionOld(lower, upper, data, N, optimal_prize);

	
	#print "\n[Agent Index, Price, Quality, Virtual Valuation, Actual Valuation]\n"
	print data; 

	#print "\n\n\n";
	print "competition list", competition_list;

	#plt.savefig('test.png');
	#plt.show();	

main();

