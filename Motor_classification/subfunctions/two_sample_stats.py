import numpy as np
import pandas as pd

from scipy import stats

from scipy.interpolate import interp1d

# Personal python functions
from subfunctions.interpretation_of_kstest import *
from subfunctions.make_a_properlist import *


# Test
# num_of_tests is the number of statistics test that you wish to use to test your null hypothesis.



def two_sample_stats(vec1, vec2, num_of_tests):
    
    # vec1 and vec2 must have at least 2 entries to process
    if len(vec1) > 2 and len(vec2) > 2:#  and vec1.all() != 0 and vec2.all() != 0:
        
        # ------------------------------
        
        # Ensure that vec1 and vec2 have the same length
        if len(vec1) > len(vec2):
            longSIG = vec1
            shortSIG = vec2
            flag = 0
        else: 
            shortSIG = vec1
            longSIG = vec2
            flag = 1
        
        # What type of signals do you need to find a statistic  for ??
        whichsig_type = 1
        if whichsig_type == 0:
            # Technique 0 : interpolation of signal (ok only for a time series)
            x = np.linspace(shortSIG[0], len(shortSIG), num=len(shortSIG), endpoint=True)
            y = shortSIG
            f = interp1d(x, y)  # default is linear interpolation
            xnew = np.linspace(shortSIG[0], len(shortSIG), num=len(longSIG), endpoint=True)
            # print('xnew : ', xnew)
            siglong = f(xnew)
        elif whichsig_type == 1:
            # Technique 0 : a processed vector (ie : values per conditions) : pad with nan
            a = len(longSIG) - len(shortSIG)
            an_array = np.empty((1,a))
            an_array[:] = np.mean(shortSIG)    # np.NaN
            fix_shSIG = shortSIG, np.ravel(an_array)
            siglong = make_a_properlist(fix_shSIG)
            print('length of fix_shSIG : ', len(siglong))
        
        # Reassign : wilcoxon needs same length vectors
        if flag == 0:
            vec1_pad = longSIG
            vec2_pad = siglong
        elif flag == 1:
            vec1_pad = siglong
            vec2_pad = longSIG
            
        # Keep original uneven vectors for ranksum, bartlett
        vec1_org = vec1
        vec2_org = vec2
           
        len_of_vecs = " ".join(str(x) for x in [len(vec1), len(vec2)])
        
        # ------------------------------
        # Check if the vectors are NOT equal each other, and they can not equal zero
        vec1_pad = np.array(vec1_pad)
        vec2_pad = np.array(vec2_pad)
        out = [i for i in range(len(vec1_pad)) if vec1_pad[i] == vec2_pad[i]]
        if len(out) == len(vec1_pad):   # if all the entries in each vector are identical
            print('vec1_pad and vec2_pad are equal')
            norm_result = np.nan
            test1 = np.nan
            pval_1 = np.nan
            test2 = np.nan
            pval_2 = np.nan
            test3 = np.nan
            pval_3 = np.nan
            len_of_vecs = " ".join(str(x) for x in [len(vec1), len(vec2)])
            # print('two_sample_stats - vec1_pad : ', vec1_pad)
            # print('two_sample_stats - vec2_pad : ', vec2_pad)
        else:
            # ------------------------------
            
            # Testing each vector with respect to the normal distribution :
            
            # statistic is the distance of the vec distribution from the normal distribution, the closer it is to zero it has a normal distribution
            # 'The D statistic is the absolute max distance (supremum) between the CDFs of the two samples. The closer this number is to 0 the more likely it is that the two samples were drawn from the same distribution. '
            
            # The p-value returned by the k-s test has the same interpretation as other p-values. You reject the null hypothesis that the two samples were drawn from the same distribution if the p-value is less than your significance level.
            Dstatistic_vec1, pvalue_vec1 = stats.kstest(vec1_org, 'norm')
            out1 = interpretation_of_kstest(Dstatistic_vec1, pvalue_vec1)
            
            Dstatistic_vec2, pvalue_vec2 = stats.kstest(vec2_org, 'norm')
            out2 = interpretation_of_kstest(Dstatistic_vec2, pvalue_vec2)
         
            # ------------------------------
         
            # Step 2 : We want to know if vec1 and vec2 have a normal distributions, 
            # if vec1 and vec2 have normal distributions, use parametric comparison tests
            # if one or both have non-normal distributions, use non-parametric comparison tests
            if out1 == 1 and out2 == 1:
                # Use parametric tests to compare the two sets :
                norm_result = 'normal'

                # Both sets have a normal distribution (parametric)  : 1) two-sample t-test, 2) Welch's t-test, 3) pooled variance t-test
                if num_of_tests == 1:
                    # 1) Two-sample t-test : mean test
                    # Compares the mean of two vectors/samples
                    stats_2sampttest, pval_1 = stats.ttest_ind(vec1_org, vec2_org, equal_var=True)
                    test1 = '2ttest'
                elif num_of_tests == 2:
                    # 1) Two-sample t-test : mean test
                    # Compares the mean of two vectors/samples
                    stats_2sampttest, pval_1 = stats.ttest_ind(vec1_org, vec2_org, equal_var=True)
                    test1 = '2ttest'
                    
                    # 2) Welch's t-test : mean test based on variance
                    # Unequal variance test to test the hypothesis that two populations have equal means 
                    stats_2welchtest, pval_2 = stats.ttest_ind(vec1_org, vec2_org, equal_var=False)
                    test2 = 'welch'
                elif num_of_tests == 3:
                    # 1) Two-sample t-test
                    # Compares the mean of two vectors/samples
                    stats_2sampttest, pval_1 = stats.ttest_ind(vec1_org, vec2_org, equal_var=True)
                    test1 = '2ttest'
                    
                    # 2) Welch's t-test : mean test based on variance
                    # Unequal variance test to test the hypothesis that two populations have equal means 
                    stats_2welchtest, pval_2 = stats.ttest_ind(vec1_org, vec2_org, equal_var=False)
                    test2 = 'welch'
                    
                    # 3) (F-test) : variance test
                    # Comparison of Variance : equivalent in R is the var.test() function for testing the assumption that the variances are the same, this is done by testing to see if the ratio of the variances is equal to 1
                    
                    # if pval_3 > alpha:
                        # Reject the null hypothesis that Var(vec1) == Var(vec2)
                    F = np.var(vec1_org) / np.var(vec2_org)
                    df1 = len(vec1_org) - 1
                    df2 = len(vec2_org) - 1
                    pval_3 = stats.f.cdf(F, df1, df2)
                    test3 = 'Ftest_2cdf'
                 
            else:
                # Use non-parametric tests to compare the two sets :
                norm_result = 'nonnormal'

                # Both sets have a non-normal distribution (non-parametric) : 1) Wilcoxon rank sum test, 2) Wilcoxon signed rank test, 3) Tarone-Ware two sampled test, 4) Friedman test
                if num_of_tests == 1:
                    # 1) Wilcoxon signed rank test : distribution test (need same length signals)
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
                    # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.*
                    w, pval_1 = stats.wilcoxon(vec1_pad, vec2_pad,  zero_method='wilcox')
                    # OR
                    # d = vec1 - vec2    # difference of vectors
                    # w, p = wilcoxon(d)
                    test1 = 'signedrank'
                elif num_of_tests == 2:
                    # 1) Wilcoxon signed rank test : distribution test
                    w, pval_1 = stats.wilcoxon(vec1_pad, vec2_pad,  zero_method='wilcox')
                    # OR
                    # d = vec1 - vec2    # difference of vectors
                    # w, p = wilcoxon(d)
                    test1 = 'signedrank'
                    
                    # 2) sum rank test : distribution test
                    Dstat, pval_2 = stats.ranksums(vec1_org, vec2_org)
                    test2 = 'sumrank'
                elif num_of_tests == 3:
                    # 1) Wilcoxon signed rank test : distribution test
                    # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero.
                    w, pval_1 = stats.wilcoxon(vec1_pad, vec2_pad,  zero_method='wilcox')
                    # OR
                    # d = vec1 - vec2    # difference of vectors
                    # w, p = wilcoxon(d)
                    test1 = 'signedrank'
                    
                    # 2) sum rank test : distribution test
                    # The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution. The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
                    Dstat, pval_2 = stats.ranksums(vec1_org, vec2_org)
                    test2 = 'sumrank'
                    
                    # 3) Bartlett's test : distribution variance test
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html
                    # Bartlett’s test tests the null hypothesis that all input samples are from populations with equal variances. For samples from significantly non-normal populations, Levene’s test levene is more robust.
                    Dstat, pval_3 = stats.bartlett(vec1_org, vec2_org)
                    test3 = 'bartlett'
                    
                elif num_of_tests == 4:
                    # 1) Wilcoxon signed rank test : distribution test
                    # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero.
                    w, pval_1 = stats.wilcoxon(vec1_pad, vec2_pad,  zero_method='wilcox')
                    # OR
                    # d = vec1 - vec2    # difference of vectors
                    # w, p = wilcoxon(d)
                    test1 = 'signedrank'
                    
                    # 2) sum rank test : distribution test
                    # The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution. The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
                    Dstat, pval_2 = stats.ranksums(vec1_org, vec2_org)
                    test2 = 'sumrank'
                    
                    # 3) Bartlett's test : distribution variance test
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html
                    # Bartlett’s test tests the null hypothesis that all input samples are from populations with equal variances. For samples from significantly non-normal populations, Levene’s test levene is more robust.
                    Dstat, pval_3 = stats.bartlett(vec1_org, vec2_org)
                    test3 = 'bartlett'
                    
                    # 4) Levene's test : variance test for non-normally distributed data
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
                    # The Levene test tests the null hypothesis that all input samples are from populations with equal variances. Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.
                    Dstat, pval_4 = stats.levene(vec1_org, vec2_org,  center='median', proportiontocut=0.05)
                    test4 = 'levene'

                    
    else:
        print('vec1 and/or vec2 are less than 2 entries')
        norm_result = np.nan
        test1 = np.nan
        pval_1 = np.nan
        test2 = np.nan
        pval_2 = np.nan
        test3 = np.nan
        pval_3 = np.nan
        len_of_vecs = " ".join(str(x) for x in [len(vec1), len(vec2)])
            
    if num_of_tests == 1:
        col0 = pd.Series(norm_result)  # string
        col1 = pd.Series(test1)  # string
        col2 = pd.Series(pval_1) # number
        col3 = pd.Series(len_of_vecs) # number
        temp = pd.concat([col0, col1, col2, col3], axis=1)
        df_res = temp.rename({0: 'norm_result', 1: 'test1', 2: 'pval_1', 3: 'lenvec1_lenvec2'}, axis=1)
    elif num_of_tests == 2:
        col0 = pd.Series(norm_result)  # string
        col1 = pd.Series(test1)  # string
        col2 = pd.Series(pval_1) # number
        col3 = pd.Series(test2)  # string
        col4 = pd.Series(pval_2) # number
        col5 = pd.Series(len_of_vecs) # number
        temp = pd.concat([col0, col1, col2, col3, col4, col5], axis=1)
        df_res = temp.rename({0: 'norm_result', 1: 'test1', 2: 'pval_1', 3: 'test2', 4: 'pval_2', 5: 'lenvec1_lenvec2'}, axis=1)
    elif num_of_tests == 3:
        # return a DataFrame, and each time you call it you can just append outputs
        col0 = pd.Series(norm_result)  # string
        col1 = pd.Series(test1)  # string
        col2 = pd.Series(pval_1) # number
        col3 = pd.Series(test2)  # string
        col4 = pd.Series(pval_2) # number
        col5 = pd.Series(test3)  # string
        col6 = pd.Series(pval_3) # number
        col7 = pd.Series(len_of_vecs) # number
        
        # Want to arrange the data in columns (stack the columns next to each other): so axis=1
        # OR think of it as the columns of the df change so you put axis=1 for columns
        temp = pd.concat([col0, col1, col2, col3, col4, col5, col6, col7], axis=1)
        df_res = temp.rename({0: 'norm_result', 1: 'test1', 2: 'pval_1', 3: 'test2', 4: 'pval_2', 5: 'test3', 6: 'pval_3', 7: 'lenvec1_lenvec2'}, axis=1)
    
    return df_res