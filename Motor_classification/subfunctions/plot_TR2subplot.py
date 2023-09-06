import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import zip_longest

from subfunctions.make_a_properlist import *
from subfunctions.compare_condi import compare_across_condi, compare_within_condi

def plot_TR2subplot(po, df_scalarmetics_exp, cou, ax, plt, exp, exp_name, anom, eval_axis_dir):

    final_temp_withincondi = pd.DataFrame()
    final_temp_acrosscondi = pd.DataFrame()

    df_long_tot = pd.DataFrame()

    outer3_name = ['rot', 'trans']

    estimator_type = 'mean'
    bonfero_thresh = 0.0167
    across_marg = 4

    num_of_tests = 3

    df = df_scalarmetics_exp[outer3_name[exp]]

    outer_name_list = ['sub', 'sup']
    inner_name_list = ['IC', 'EC', 'NC', 'NR']

    ax_all_vals_exp = []
    for i in range(3):  # 0=RO/LR, 1=PI/FB, 2=YA/UD
        print('axis: ', i)
        
        ax_all_vals = []
        for j in range(1,3):  # 1=sub, 2=sup
            print('sub/sup: ', j)
            
            num_of_sub = len(df.subject.value_counts().to_numpy())
            # print('num_of_sub: ', num_of_sub)
            
            
            if eval_axis_dir == 'all_ax':
                # ------------------------------------
                # Combined axis direction evaluation 
                # ------------------------------------
                # 1a) I need the counts per subject to get the population count for the error bars
                IC_vals = list(df.TR[(df.res_type == 1) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy())
                
                # More than one or category requires that you sum the entries
                temp0_0 = list(df.TR[(df.res_type == 2) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy())
                print('res_type=2: ', temp0_0)
                temp0_1 = list(df.TR[(df.res_type == 4) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy())
                print('res_type=4: ', temp0_1)
                temp0_2 = list(df.TR[(df.res_type == 5) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy())
                print('res_type=5: ', temp0_2)
                
                applist = temp0_0, temp0_1, temp0_2
                EC_vals = make_a_properlist(applist)
                print('EC_vals: ', EC_vals)

                # More than one or category requires that you sum the entries
                temp1_0 = list(df.TR[(df.res_type == 3) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy())
                print('res_type=3: ', temp1_0)
                temp1_1 = list(df.TR[(df.res_type == 6) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy())
                print('res_type=6: ', temp1_1)
                temp1_2 = list(df.TR[(df.res_type == 7) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy())
                print('res_type=7: ', temp1_2)
                
                applist = temp1_0, temp1_1, temp1_2
                NC_vals = make_a_properlist(applist)
                print('NC_vals: ', NC_vals)
                
                # No Response is zero or nan because there was no response : do not even include it in the final plot
                # NR_vals = df.TR[(df.res_type == 9) & (df.ax == i) & (np.abs(df.ss) == j)].to_numpy()
            elif eval_axis_dir == 'pos_ax':
                # ------------------------------------
                # POSITIVE axis direction evaluation 
                # ------------------------------------
                # 1a) I need the counts per subject to get the population count for the error bars
                IC_vals = list(df.TR[(df.res_type == 1) & (df.ax == i) & (df.ss == j)].to_numpy())
                
                # More than one or category requires that you sum the entries
                temp0_0 = list(df.TR[(df.res_type == 2) & (df.ax == i) & (df.ss == j)].to_numpy())
                print('res_type=2: ', temp0_0)
                temp0_1 = list(df.TR[(df.res_type == 4) & (df.ax == i) & (df.ss == j)].to_numpy())
                print('res_type=4: ', temp0_1)
                temp0_2 = list(df.TR[(df.res_type == 5) & (df.ax == i) & (df.ss == j)].to_numpy())
                print('res_type=5: ', temp0_2)
                
                applist = temp0_0, temp0_1, temp0_2
                EC_vals = make_a_properlist(applist)
                print('EC_vals: ', EC_vals)

                # More than one or category requires that you sum the entries
                temp1_0 = list(df.TR[(df.res_type == 3) & (df.ax == i) & (df.ss == j)].to_numpy())
                print('res_type=3: ', temp1_0)
                temp1_1 = list(df.TR[(df.res_type == 6) & (df.ax == i) & (df.ss == j)].to_numpy())
                print('res_type=6: ', temp1_1)
                temp1_2 = list(df.TR[(df.res_type == 7) & (df.ax == i) & (df.ss == j)].to_numpy())
                print('res_type=7: ', temp1_2)
                
                applist = temp1_0, temp1_1, temp1_2
                NC_vals = make_a_properlist(applist)
                print('NC_vals: ', NC_vals)
                
                # No Response is zero or nan because there was no response : do not even include it in the final plot
                # NR_vals = df.TR[(df.res_type == 9) & (df.ax == i) & (df.ss == j)].to_numpy()
            elif eval_axis_dir == 'neg_ax':
                # ------------------------------------
                # NEGATIVE axis direction evaluation 
                # ------------------------------------
                # 1a) I need the counts per subject to get the population count for the error bars
                IC_vals = list(df.TR[(df.res_type == 1) & (df.ax == i) & (df.ss == -j)].to_numpy())
                
                # More than one or category requires that you sum the entries
                temp0_0 = list(df.TR[(df.res_type == 2) & (df.ax == i) & (df.ss == -j)].to_numpy())
                print('res_type=2: ', temp0_0)
                temp0_1 = list(df.TR[(df.res_type == 4) & (df.ax == i) & (df.ss == -j)].to_numpy())
                print('res_type=4: ', temp0_1)
                temp0_2 = list(df.TR[(df.res_type == 5) & (df.ax == i) & (df.ss == -j)].to_numpy())
                print('res_type=5: ', temp0_2)
                
                applist = temp0_0, temp0_1, temp0_2
                EC_vals = make_a_properlist(applist)
                print('EC_vals: ', EC_vals)

                # More than one or category requires that you sum the entries
                temp1_0 = list(df.TR[(df.res_type == 3) & (df.ax == i) & (df.ss == -j)].to_numpy())
                print('res_type=3: ', temp1_0)
                temp1_1 = list(df.TR[(df.res_type == 6) & (df.ax == i) & (df.ss == -j)].to_numpy())
                print('res_type=6: ', temp1_1)
                temp1_2 = list(df.TR[(df.res_type == 7) & (df.ax == i) & (df.ss == -j)].to_numpy())
                print('res_type=7: ', temp1_2)
                
                applist = temp1_0, temp1_1, temp1_2
                NC_vals = make_a_properlist(applist)
                print('NC_vals: ', NC_vals)
                
                # No Response is zero or nan because there was no response : do not even include it in the final plot
                # NR_vals = df.TR[(df.res_type == 9) & (df.ax == i) & (df.ss == -j)].to_numpy()
            
            
            # Put desired values in a DataFrame
            # all_vals = [IC_vals, EC_vals, NC_vals, NR_vals]
            all_vals = [IC_vals, EC_vals, NC_vals]
            longcol_num = []
            longcol_text = []
            for n1 in range(len(all_vals)):
                for n2 in range(len(all_vals[n1])):
                    longcol_num.append(all_vals[n1][n2])
                    longcol_text.append(inner_name_list[n1])

            col0 = pd.Series(np.ravel(longcol_text))
            col1 = pd.Series(np.ravel(longcol_num))
            temp = pd.concat([col0, col1], axis=1)
            df_long = temp.rename({0: 'str', 1: 'vals'}, axis=1)
            
            # Make an appended df_long across all conditions for statistical processing
            df_long_copy = temp.rename({0: 'str_%s_%s_%s' % (outer3_name[exp], anom[i], outer_name_list[j-1]), 1: 'vals_%s_%s_%s' % (outer3_name[exp], anom[i], outer_name_list[j-1])}, axis=1)
            df_long_tot = pd.concat([df_long_tot, df_long_copy], axis=1)
            # ----------------
            
            # Plot each subORsup per axis (for now)
            # sns.set(font_scale = 1.7, style="white", palette=None) 
            # OR
            sns.set(font_scale = 2) # default is without style and palette
            
            sns.color_palette("light:#90a4ae", as_cmap=True)  # Greys_d, light:#5A9
            
            if i == 0:
                sns.barplot(x="str", y="vals", data=df_long, ax=ax[po,cou], ci="sd", capsize=.2, estimator=np.mean, palette="light:#90a4ae", errcolor="#3E26A8", linewidth=3, edgecolor="#3E26A8")
            elif i == 1:
                sns.barplot(x="str", y="vals", data=df_long, ax=ax[po,cou], ci="sd", capsize=.2, estimator=np.mean, palette="light:#90a4ae", errcolor="#02BAC3", linewidth=3, edgecolor="#02BAC3")
            elif i == 2:
                sns.barplot(x="str", y="vals", data=df_long, ax=ax[po,cou], ci="sd", capsize=.2, estimator=np.mean, palette="light:#90a4ae", errcolor="#F6EF1F", linewidth=3, edgecolor="#F6EF1F")
            ax[po,cou].set(ylim=(0, 35))
            
            # ax[po,0].set_ylabel('Normalized trial count (%s)' % (exp_name))
            ax[po,0].set_ylabel('Mean RT (%s)' % (exp_name), fontsize=22)
            
            if cou > 0:
                ax[po,cou].set_ylabel(' ')
                ax[po,cou].set_yticks([])
            
            if j == 1:
                ax[po,cou].set_xlabel('%s sub' % (anom[i]), fontsize=22)
            elif j == 2:
                ax[po,cou].set_xlabel('%s sup' % (anom[i]), fontsize=22)
                
            ax_all_vals.append(all_vals)
            
            cou = cou + 1
            plt.xticks(fontsize=22)
            
            
    # ----------------
    # Statistics
    # ----------------
    # Trying a different way : because across conditions need all the data
    # 1) put needed data in a DataFrame, 2) read from DataFrame and do stats at the end:
    if exp == 0:
        # Within axes : differences with axis across same condition (Visually looked for any combinations) 
        which_anom = 'RO'
        incon = 'IC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'EC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'NC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        
        which_anom = 'PI'
        incon = 'IC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'EC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'NC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        
        which_anom = 'YA'
        incon = 'IC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'EC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'NC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        
        # ---------
        
        # Across axes : differences between axis but same condition (Visually looked for any combinations)
        incon = 'IC'
        ss = 'sub'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'EC'
        ss = 'sub'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'NC'
        ss = 'sub'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        # ---------
        
        incon = 'IC'
        ss = 'sup'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'EC'
        ss = 'sup'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'NC'
        ss = 'sup'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
    elif exp == 1:
        # Within axes : differences with axis across same condition (Visually looked for any combinations) 
        which_anom = 'LR'
        incon = 'IC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'EC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'NC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        
        which_anom = 'FB'
        incon = 'IC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'EC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'NC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        
        which_anom = 'UD'
        incon = 'IC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'EC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        incon = 'NC'
        print(exp, which_anom, incon)
        final_temp_withincondi = compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type)
        
        # ---------
        
        # Across axes : differences between axis but same condition (Visually looked for any combinations)
        incon = 'IC'
        ss = 'sub'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'EC'
        ss = 'sub'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'NC'
        ss = 'sub'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        # ---------
        
        incon = 'IC'
        ss = 'sup'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'EC'
        ss = 'sup'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)
        
        incon = 'NC'
        ss = 'sup'
        print(exp, ss, incon)
        final_temp_acrosscondi = compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type)

        plt.xticks(fontsize=22)

    return final_temp_withincondi, final_temp_acrosscondi, df_long_tot