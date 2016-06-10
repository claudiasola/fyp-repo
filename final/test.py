import numpy as np
import scipy.stats as stats

def main():

	#data = np.load('agentList.npy', 'r');
	#unironed = np.load('unironedVB.npy', 'r');
	#bound = np.load('boundaries.npy', 'r');
	ora = np.load('profitORA.npy', 'r');
	era = np.load('profitERA.npy', 'r');
	a = np.zeros(2);
	b = np.ones(2);

	#fpc = np.load('profitFPC.npy', 'r');	
	u_value, p_value = stats.mannwhitneyu(ora, era);
#	u_value, p_value = stats.mannwhitneyu(a, b);
	print len(ora);
	print u_value, p_value;
	#49.0 0.484924988497

main();