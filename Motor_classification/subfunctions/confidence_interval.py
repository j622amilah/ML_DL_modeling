import numpy as np
import scipy.stats


def confidence_interval(data, desired_CI=0.95):

    # General procedure:
    mean_dat = np.mean(data)  # sample mean
    std_dat = np.std(data)

    # Step 1 : degrees of freedom = sample size - 1 
    dof = len(data)-1

    # ----------------------------------------------

    # Step 2: Obtain the t-distribution (some people divide alpha by 2 - it sets the lower and upper tails further from the mean)
    # alpha = (1 - desired_CI)   # (alpha + CL = 1)
    # cv = scipy.stats.t.ppf(alpha/2, dof)  # critical value on t-distribution table

    # OR

    # Step 2: Obtain the t-distribution
    # Statisticians use tα to represent the t statistic that has a cumulative probability of (1 - α). For example, suppose we were interested in the t statistic having a cumulative probability of 0.95. In this example, α would be equal to (1 - 0.95) or 0.05. We would refer to the t statistic as t0.05.

    # The value of t0.05 depends on the number of degrees of freedom. For example, with 2 degrees of freedom, t_0.05 is equal to 2.92; but with 20 degrees of freedom, t_0.05 is equal to 1.725.

    # Note: Because the t distribution is symmetric about a mean of zero, the following is true.

    # t_α = -t_1 - alpha       And       t1 - alpha = -t_α

    # Thus, if t_0.05 = 2.92, then t_0.95 = -2.92.

    # Both are the same because the t distribution is symmetric 
    # alpha = (1 - desired_CI)   # (alpha + CL = 1)
    # cv = scipy.stats.t.ppf(alpha, dof)   # critical value on t-distribution table
    # OR
    cv = scipy.stats.t.ppf(desired_CI, dof)   # critical value on t-distribution table


    t_crit = np.abs(cv)  # take the absolute value, because the table is symmetric
    # ----------------------------------------------

    # Step 3: Standard error of the mean
    sem = std_dat/np.sqrt(len(data)) 

    # t_crit is like a scaling factor to shift the lower_tail and upper_tail depending on
    # what you choose to be significant based on the desired_CI
    margin = t_crit*sem 

    # the lower end of the range
    lower_tail = mean_dat - margin
    # print('lower_tail : ' + str(lower_tail))

    # the upper end of the range
    upper_tail = mean_dat + margin
    # print('upper_tail : ' + str(upper_tail))

    # ----------------------------------------------


    # OR you can use numpy
    # alpha = (1 - desired_CI)   # (alpha + CL = 1)
    # lower_tail, upper_tail = np.percentile(data, [100*alpha, 100*desired_CI])
    
    return mean_dat, lower_tail, upper_tail