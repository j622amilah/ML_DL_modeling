import numpy as np
from scipy import stats

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import QuantileTransformer

# import seaborn as sns
# import matplotlib.pyplot as plt

from subfunctions.interpretation_of_kstest import *


def normal_distribution_feature_data(feat, plotORnot):
    # Make distribution of feature parametric per trial
    
    
    # Check for constant vectors
    boolval = [isnan(feat[i]) for i in range(len(feat))]
    if any(boolval) == False:
    	return feat
    else:
		baseline_f = feat - feat[0]
		if int(np.mean(baseline_f)) == 0:
		    return feat
		else:
		    # Check if the feature has a normal distribution

		    statistic, pvalue = stats.kstest(feat, 'norm')
		    # print('statistic : ', statistic, ', pvalue : ', pvalue)
		    result = interpretation_of_kstest(statistic, pvalue)
		    # print('result : ', result)

		    # Initialize : use feature if it has normal distribution OR if normal distribution can not be found
		    norm_feat = feat

		    if result == 0 and np.isnan(feat).any() == False:
		        # Does not work for negative values, so shift up a little bit above zero
		        pos_shift = feat - (np.min(feat)-0.000001)
		        # print('min val : ', np.min(pos_shift))

		        # ----------------

		        normaldist0 = stats.boxcox(pos_shift)
		        # ----------------
		        # OR
		        # ----------------
		        # Stack two signals because the functions can not process a single signal alone
		        X = pos_shift, pos_shift
		        X = np.transpose(X)
		        # print('shape of X : ', X.shape)
		        # ----------------

		        # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html

		        # ----------------
		        bc = PowerTransformer(method='box-cox')
		        normaldist1 = bc.fit(X).transform(X)
		        # ----------------

		        # ----------------
		        yj = PowerTransformer(method='yeo-johnson')
		        normaldist2 = yj.fit(X).transform(X)
		        # ----------------

		        # ----------------

		        rng = np.random.RandomState(0)
		        num_of_samps = len(feat)
		        qt = QuantileTransformer(n_quantiles=num_of_samps, output_distribution='normal', random_state=rng)
		        normaldist3 = qt.fit(X).transform(X)
		        # ----------------

		        # Some of the distribution transformations do not always work.  Test to see if the test suceeded.
		        statistic, pvalue = stats.kstest(normaldist0[0], 'norm')
		        result0 = interpretation_of_kstest(statistic, pvalue)

		        statistic, pvalue = stats.kstest(normaldist1[:,0], 'norm')
		        result1 = interpretation_of_kstest(statistic, pvalue)

		        statistic, pvalue = stats.kstest(normaldist2[:,0], 'norm')
		        result2 = interpretation_of_kstest(statistic, pvalue)

		        statistic, pvalue = stats.kstest(normaldist3[:,0], 'norm')
		        result3 = interpretation_of_kstest(statistic, pvalue)

		        allres = [result0, result1, result2, result3]
		        # print('allres : ', allres)

		        all_dat = normaldist0[0], list(normaldist1[:,0]), list(normaldist2[:,0]), list(normaldist3[:,0])
		        all_dat = np.transpose(all_dat)

		        for i in range(len(allres)):
		            if allres[i] == 1:
		                norm_feat = all_dat[:,i]
		                break

		        # if plotORnot == 1:
		            # # histogram
		            # print('shape of all_dat : ', all_dat.shape)
		            # dfout = pd.DataFrame(data=all_dat)


		            # fig, ax=plt.subplots(4,1)
		            # sns.distplot(dfout[0], ax=ax[0], bins=30, label="stats") 
		            # sns.distplot(dfout[1], ax=ax[1], bins=30, label='bc')
		            # sns.distplot(dfout[2], ax=ax[2], bins=30, label='yj')
		            # sns.distplot(dfout[3], ax=ax[3], bins=30, label='qt')

		    return norm_feat
