from __future__ import division
from scipy.stats import norm
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

precision_variable = 0.00001;

def computeVirtualValuation(price, quality):
	ironed_value = findIronedValue(price);
	virtual_valuation = quality + ironed_value;

	return virtual_valuation;

def computeActualValuation(price, quality):	
	actual_valuation = quality - price;

	return actual_valuation;

def findIronedValue(price):
	count = 0;
	value_set = False;
	x_index = int(price/precision_variable) - int(lower/precision_variable);
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

def computeOptimalReverseAuction(data, N):
	#print "\nOptimal Reverse Auction\n"

	virtual_valuation = np.full([N,1, ],0);
	for i in range (0,N):
		#data[i][3] = computeVirtualValuation(data[i][1], data[i][2]);
		virtual_valuation[i] = computeVirtualValuation(data[i][1], data[i][2]);

	#determine optimal value of reserve parameters
	price_reserve = 3.5;
	quality_reserve = 7;
	valuation_reserve = computeVirtualValuation(price_reserve, quality_reserve);

	information_rent, utility_of_principal, prize = 0, 0, 0; 
	valuation_b, b_index = 0, 0;
	valuation_sb, sb_index = 0, 0;
	
	for i in range (0,N):
		valuation_temp = virtual_valuation[i];
		if((valuation_temp > valuation_b) or (valuation_temp == valuation_b and data[i][1] < data[b_index][1])):
			valuation_sb = valuation_b;
			sb_index = b_index;
			valuation_b = valuation_temp;
			b_index = i;
		elif(valuation_temp > valuation_sb):
			valuation_sb = valuation_temp;
			sb_index = i;
	
	#print "Best:", b_index;
	#print "\twith virtual valuation", valuation_b;
	#print "2nd Best:", sb_index;
	#print "\twith virtual valuation", valuation_sb;
	
	if(valuation_b >= valuation_reserve):
		prize = binarySearchAlgorithm(valuation_sb,data[b_index][2], computeVirtualValuation);
		#print "\nPrize:", prize;
		#print "\tWinner:", b_index;
		information_rent = prize - data[b_index][1]; 
		utility_of_principal = data[b_index][2] - prize;
		
	#else: 
		#print "\tNo Winner";

	#print "\tInformation Rent:", information_rent;
	#print "\tUtility of Principal:", utility_of_principal;	

	return utility_of_principal;

def computeEfficientReverseAuction(data, N):
	#print "\nEfficient Reverse Auction\n"

	#determine optimal value of reserve parameters
	price_reserve = 3.5;
	quality_reserve = 7;
	valuation_reserve = computeActualValuation(price_reserve, quality_reserve);

	information_rent, utility_of_principal, prize = 0, 0, 0; 
	valuation_b, b_index = 0, 0;
	valuation_sb, sb_index = 0, 0;
	
	for i in range (0,N):
		#data[i][4] = computeActualValuation(data[i][1], data[i][2]);
		#valuation_temp = data[i][4];
		valuation_temp = computeActualValuation(data[i][1], data[i][2]);
		if((valuation_temp > valuation_b) or (valuation_temp == valuation_b and data[i][1] < data[b_index][1])):
			valuation_sb = valuation_b;
			sb_index = b_index;
			valuation_b = valuation_temp;
			b_index = i;
		elif(valuation_temp > valuation_sb):
			valuation_sb = valuation_temp;
			sb_index = i;
	
	#print "Best:", b_index;
	#print "\twith actual valuation", valuation_b;
	#print "2nd Best:", sb_index;
	#print "\twith actual valuation", valuation_sb;

	if(valuation_b >= valuation_reserve):
		prize = binarySearchAlgorithm(valuation_sb,data[b_index][2], computeActualValuation);
		#print "\nPrize:", prize;
		#print "\tWinner:", b_index;
		information_rent = prize - data[b_index][1]; 
		utility_of_principal = data[b_index][2] - prize;
		
	#else: 
		#print "\tNo Winner";

	#print "\tInformation Rent:", information_rent;
	#print "\tUtility of Principal:", utility_of_principal;	

	return utility_of_principal;

def binarySearchAlgorithm(valuation_sb, b_quality, functionName):
	#print "\nBINARY SEARCH\n"
	error1, error2 = 1, 1;
	count = 0;
	lower_temp = lower;
	upper_temp = upper;

	#while (error != 0):
	while (np.absolute(error1) > precision_variable and np.absolute(error2) > precision_variable):
	
		mid = (lower_temp + upper_temp)/2;
		valuation_bmax = functionName(mid, b_quality);

		if valuation_bmax < valuation_sb:
			upper_temp = mid;
		elif valuation_bmax > valuation_sb:
			lower_temp = mid;

		error1 = valuation_bmax - valuation_sb;
		error2 = upper_temp - lower_temp;
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

def selectOptimalPrize(competition_results):
	max_utility = 0;
	
	for i in range(0, len(competition_results)):
		if(competition_results[i][1] > max_utility):
			max_utility = competition_results[i][1];
			optimal_prize =  competition_results[i][0];
 
 	#print "\n\nOptimal prize is\n\n", optimal_prize;
	return optimal_prize;

def findOptimalFixedPriceCompetitionPrize(competition_list, competition_number, price_number):
	#price_number = 100;
	#price_number = (upper-lower)/(2*len(competition_list));
	step = (upper - lower)/price_number;
	competition_results = [];
	c_index = 0;

	for p in frange(lower, upper, step): 
		
		total_utility = 0;
		#print "\nPrize:", p;
	
		for i in range(0, competition_number):
			winner_utility = computeFixedPriceCompetitionMultiple(competition_list[i], p);
			total_utility += winner_utility;

		average_utility = total_utility/competition_number;
		#print "\tAverage Utility: ", average_utility;

		competition_results.append([]);
		competition_results[c_index].append(p);
		competition_results[c_index].append(average_utility);
		c_index += 1;

	#print "average utility per price", competition_results;
	optimal_prize = selectOptimalPrize(competition_results);

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
	#print "\nFixed Price Competition\n"

	winner_index = np.nan;
	max_quality, information_rent, utility = 0, 0, 0;

	#print "Prize:", prize;	

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
		#print "\tWinner:", winner_index;

	#else:
	#	print "\tNo Winner"

	#print "\tInformation Rent:", information_rent;
	#print "\tUtility of Principal:", utility;	

	return utility;

def runCompetitionSchemes(data, N, optimal_prize):
	profit_ORA = computeOptimalReverseAuction(data, N);
	profit_ERA = computeEfficientReverseAuction(data, N);
	profit_FPC = computeFixedPriceCompetition(data, optimal_prize);

	return profit_ORA, profit_ERA, profit_FPC;

def formatIntoCSV(results):
	string ='';
	count = 0;
	
	while (count < len(results)-1):
		string += str(results[count]) + ",";
		count += 1;

	string += str(results[count]);

	return string;

def main():

	global lower, upper;

	N = int(sys.argv[1]); 
	seed = int(sys.argv[2]);	
	lower = float(sys.argv[3]); # lower boundary of truncated normal distribution
	upper = float(sys.argv[4]); # upper boundary of truncated normal distribution
	competition_number = int(sys.argv[5]); 

	setRandomSeed(seed);

	global unironed_baseValuation, boundaries_to_iron;
	unironed_baseValuation = np.load('unironedVB.npy', 'r');
	boundaries_to_iron = np.load('boundaries.npy', 'r');

	competition_list = np.load('agentList.npy', 'r');

	optimal_prize = findOptimalFixedPriceCompetitionPrize(competition_list, competition_number, 100);
	
	total_profit_ORA, total_profit_ERA, total_profit_FPC = 0,0,0;
	profit_ORA = np.full([competition_number,1, ],0);
	profit_ERA = np.full([competition_number,1, ],0);
	profit_FPC = np.full([competition_number,1, ],0);

	for i in range(0, competition_number):
		data = competition_list[i];
		#print "\n\nCompetition Round", i+1, "\n";
		profit_ORA[i], profit_ERA[i], profit_FPC[i] = runCompetitionSchemes(data, N, optimal_prize);
		total_profit_ORA += profit_ORA[i][0];
		total_profit_ERA += profit_ERA[i][0];
		total_profit_FPC += profit_FPC[i][0];

	expected_profit_ORA = total_profit_ORA / competition_number;
	expected_profit_ERA = total_profit_ERA / competition_number;
	expected_profit_FPC = total_profit_FPC / competition_number;

	ora = file('profitORA.npy', 'w');
	np.save(ora, profit_ORA);
	era = file('profitERA.npy', 'w');
	np.save(era, profit_ERA);

	total_var_ORA, total_var_ERA, total_var_FPC = 0,0,0;
	var_ORA = [];
	var_ERA = [];
	var_FPC = [];

	for i in range(0, competition_number):
		var_ORA.insert(i, np.power(profit_ORA[i][0] - expected_profit_ORA, 2) * (competition_number / (competition_number -1)));
		total_var_ORA += var_ORA[i];
		var_ERA.insert(i, np.power(profit_ERA[i][0] - expected_profit_ERA, 2) * (competition_number / (competition_number -1)));
		total_var_ERA += var_ERA[i];
		var_FPC.insert(i, np.power(profit_FPC[i][0] - expected_profit_FPC, 2) * (competition_number / (competition_number -1)));
		total_var_FPC += var_FPC[i];

	total_var_profit_ORA = total_var_ORA / competition_number
	total_var_profit_ERA = total_var_ERA / competition_number;
	total_var_profit_FPC = total_var_FPC / competition_number;
	results = [];
	results.insert(0, competition_number);
	results.insert(1, seed);
	results.insert(2, expected_profit_ORA);
	results.insert(3, total_var_profit_ORA);
	results.insert(4, expected_profit_ERA);
	results.insert(5, total_var_profit_ERA);
	results.insert(6, expected_profit_FPC);
	results.insert(7, total_var_profit_FPC);
	
	csv = formatIntoCSV(results);
	print csv;

	#print competition_list;

main();

