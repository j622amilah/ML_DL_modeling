import pandas as pd

from subfunctions.two_sample_stats import *


def get_ax_for_across_condi(compare_name, outer_name_list, anom, stat_type):
    # Break input into word pieces
    compare_name2 = compare_name.split()
    # print('compare_name2 : ', compare_name2)

    # Calculate ind_outer1
    # 1) which word in compare_name2 is in outer_name_list
    print('compare_name2 : ', compare_name2)
    print('outer_name_list : ', outer_name_list)
    
    for i in compare_name2:
        for j in range(len(outer_name_list)):
            if i == outer_name_list[j]:
                ind_outer1 = j
    

    # In the given order compare_name2[2] is ax_ind0
    # we need to calculate ind_outer2 for the entry in compare_name2[2]

    if stat_type == 'across':
        for j in range(len(anom)):
            if compare_name2[2] == anom[j]:
                ind_outer2 = j
                
        # print('ind_outer2 : ', ind_outer2)
        ax_ind0 = len(outer_name_list)*ind_outer2 + ind_outer1
        # print('ax_ind0 : ', ax_ind0)

        for j in range(len(anom)):
            if compare_name2[3] == anom[j]:
                ind_outer2 = j
        # print('ind_outer2 : ', ind_outer2)
        ax_ind1 = len(outer_name_list)*ind_outer2 + ind_outer1
        # print('ax_ind1 : ', ax_ind1)
    elif stat_type == 'within':
        ind_outer1 = [0, 1]
    
        for j in range(len(anom)):
            if compare_name2[1] == anom[j]:
                ind_outer2 = j
        
        ax_ind=[]
        for qq in ind_outer1:
            ax_ind.append(len(outer_name_list)*ind_outer2 + qq)
        # print('ax_ind : ', ax_ind)
        ax_ind0 = ax_ind[0]
        ax_ind1 = ax_ind[1]
        
    return ax_ind0, ax_ind1
    


def compare_across_condi(df_long_tot, incon, num_of_tests, final_temp_acrosscondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, ss, exp, estimator_type):
    
    fsize = 22
    rshift = 0.22
    stat_type = 'across'
    
    percentage = 0.02
    
    if exp == 0:
        if ss == 'sub':
            vec0 = df_long_tot.vals_rot_RO_sub[(df_long_tot.str_rot_RO_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_rot_PI_sub[(df_long_tot.str_rot_PI_sub == incon)].to_numpy()
            vec2 = df_long_tot.vals_rot_YA_sub[(df_long_tot.str_rot_YA_sub == incon)].to_numpy()
        elif ss == 'sup':
            vec0 = df_long_tot.vals_rot_RO_sup[(df_long_tot.str_rot_RO_sup == incon)].to_numpy()
            vec1 = df_long_tot.vals_rot_PI_sup[(df_long_tot.str_rot_PI_sup == incon)].to_numpy()
            vec2 = df_long_tot.vals_rot_YA_sup[(df_long_tot.str_rot_YA_sup == incon)].to_numpy()
    elif exp == 1:
        if ss == 'sub':
            vec0 = df_long_tot.vals_trans_LR_sub[(df_long_tot.str_trans_LR_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_trans_FB_sub[(df_long_tot.str_trans_FB_sub == incon)].to_numpy()
            vec2 = df_long_tot.vals_trans_UD_sub[(df_long_tot.str_trans_UD_sub == incon)].to_numpy()
        elif ss == 'sup':
            vec0 = df_long_tot.vals_trans_LR_sup[(df_long_tot.str_trans_LR_sup == incon)].to_numpy()
            vec1 = df_long_tot.vals_trans_FB_sup[(df_long_tot.str_trans_FB_sup == incon)].to_numpy()
            vec2 = df_long_tot.vals_trans_UD_sup[(df_long_tot.str_trans_UD_sup == incon)].to_numpy()
    
    # -----------
    # vec0 and vec1 compare
    # -----------
    # print('compare_across_condi - vec0 : ', vec0)
    # print('compare_across_condi - vec1 : ', vec1)
    
    # Statistical tests
    df_res = two_sample_stats(vec0, vec1, num_of_tests)
    compare_name = '%s %s %s %s' % (incon, ss, anom[0], anom[1])
    # Saving stats in a DataFrame
    col0 = pd.Series('%s' % (compare_name))  # string
    temp_cur = pd.concat([col0, df_res], axis=1)
    final_temp_acrosscondi = pd.concat([temp_cur, final_temp_acrosscondi], axis=0)
    
    # Plotting stars
    if df_res.pval_1.to_numpy()[0] < bonfero_thresh and df_res.pval_2.to_numpy()[0] < bonfero_thresh:
        row_ind = [i for i in range(len(inner_name_list)) if incon == inner_name_list[i]][0]
        ax_ind0, ax_ind1 = get_ax_for_across_condi(compare_name, outer_name_list, anom, stat_type)
        
        if estimator_type == 'sum':
            ax[po, ax_ind0].text(row_ind-rshift, np.sum(vec0)+across_marg, "**", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.sum(vec1)+across_marg, "**", fontsize=fsize)
        elif estimator_type == 'mean':
            across_marg0 = np.max(vec0)*percentage
            across_marg1 = np.max(vec1)*percentage
            ax[po, ax_ind0].text(row_ind-rshift, np.max(vec0)+across_marg0, "**", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.max(vec1)+across_marg1, "**", fontsize=fsize)
    
    
    # -----------
    # vec0 and vec2 compare
    # -----------
    # print('compare_across_condi - vec0 : ', vec0)
    # print('compare_across_condi - vec2 : ', vec2)
    
    df_res = two_sample_stats(vec0, vec2, num_of_tests)
    compare_name = '%s %s %s %s' % (incon, ss, anom[0], anom[2])
    # Saving stats in a DataFrame
    col0 = pd.Series('%s' % (compare_name))  # string
    temp_cur = pd.concat([col0, df_res], axis=1)
    final_temp_acrosscondi = pd.concat([temp_cur, final_temp_acrosscondi], axis=0)
    
    # Plotting stars
    if df_res.pval_1.to_numpy()[0] < bonfero_thresh and df_res.pval_2.to_numpy()[0] < bonfero_thresh:
        row_ind = [i for i in range(len(inner_name_list)) if incon == inner_name_list[i]][0]
        ax_ind0, ax_ind1 = get_ax_for_across_condi(compare_name, outer_name_list, anom, stat_type)
        
        if estimator_type == 'sum':
            ax[po, ax_ind0].text(row_ind-rshift, np.sum(vec0)+across_marg, "**", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.sum(vec2)+across_marg, "**", fontsize=fsize)
        elif estimator_type == 'mean':
            across_marg0 = np.max(vec0)*percentage
            across_marg1 = np.max(vec2)*percentage
            ax[po, ax_ind0].text(row_ind-rshift, np.max(vec0)+across_marg0, "**", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.max(vec2)+across_marg1, "**", fontsize=fsize)


    # -----------
    # vec1 and vec2 compare
    # -----------
    # print('compare_across_condi - vec1 : ', vec1)
    # print('compare_across_condi - vec2 : ', vec2)
    
    df_res = two_sample_stats(vec1, vec2, num_of_tests)
    compare_name = '%s %s %s %s' % (incon, ss, anom[1], anom[2])
    # Saving stats in a DataFrame
    col0 = pd.Series('%s' % (compare_name))  # string
    temp_cur = pd.concat([col0, df_res], axis=1)
    final_temp_acrosscondi = pd.concat([temp_cur, final_temp_acrosscondi], axis=0)
    
    # Plotting stars
    if df_res.pval_1.to_numpy()[0] < bonfero_thresh and df_res.pval_2.to_numpy()[0] < bonfero_thresh:
        row_ind = [i for i in range(len(inner_name_list)) if incon == inner_name_list[i]][0]
        ax_ind0, ax_ind1 = get_ax_for_across_condi(compare_name, outer_name_list, anom, stat_type)
        
        if estimator_type == 'sum':
            ax[po, ax_ind0].text(row_ind-rshift, np.sum(vec1)+across_marg, "**", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.sum(vec2)+across_marg, "**", fontsize=fsize)
        elif estimator_type == 'mean':
            across_marg0 = np.max(vec1)*percentage
            across_marg1 = np.max(vec2)*percentage
            ax[po, ax_ind0].text(row_ind-rshift, np.max(vec1)+across_marg0, "**", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.max(vec2)+across_marg1, "**", fontsize=fsize)
    
    
    return final_temp_acrosscondi
    
    
    
    
    
def compare_within_condi(df_long_tot, incon, num_of_tests, final_temp_withincondi, bonfero_thresh, inner_name_list, across_marg, outer_name_list, anom, ax, po, exp, which_anom, estimator_type):

    fsize = 22
    rshift = 0.1
    stat_type = 'within'

    if exp == 0:
        if which_anom == 'RO':
            vec0 = df_long_tot.vals_rot_RO_sub[(df_long_tot.str_rot_RO_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_rot_RO_sup[(df_long_tot.str_rot_RO_sup == incon)].to_numpy()
        elif which_anom == 'PI':
            vec0 = df_long_tot.vals_rot_PI_sub[(df_long_tot.str_rot_PI_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_rot_PI_sup[(df_long_tot.str_rot_PI_sup == incon)].to_numpy()
        elif which_anom == 'YA':
            vec0 = df_long_tot.vals_rot_YA_sub[(df_long_tot.str_rot_YA_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_rot_YA_sup[(df_long_tot.str_rot_YA_sup == incon)].to_numpy()
    elif exp == 1:
        if which_anom == 'LR':
            vec0 = df_long_tot.vals_trans_LR_sub[(df_long_tot.str_trans_LR_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_trans_LR_sup[(df_long_tot.str_trans_LR_sup == incon)].to_numpy()
        elif which_anom == 'FB':
            vec0 = df_long_tot.vals_trans_FB_sub[(df_long_tot.str_trans_FB_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_trans_FB_sup[(df_long_tot.str_trans_FB_sup == incon)].to_numpy()
        elif which_anom == 'UD':
            vec0 = df_long_tot.vals_trans_UD_sub[(df_long_tot.str_trans_UD_sub == incon)].to_numpy()
            vec1 = df_long_tot.vals_trans_UD_sup[(df_long_tot.str_trans_UD_sup == incon)].to_numpy()
    # print('compare_within_condi - vec0 : ', vec0)
    # print('compare_within_condi - vec1 : ', vec1)
    
    # -----------
    # vec0 and vec1 compare
    # -----------
    # Statistical tests
    df_res = two_sample_stats(vec0, vec1, num_of_tests)
    compare_name = '%s %s %s %s' % (incon, which_anom, outer_name_list[0], outer_name_list[1])
    # Saving stats in a DataFrame
    col0 = pd.Series('%s' % (compare_name))  # string
    temp_cur = pd.concat([col0, df_res], axis=1)
    final_temp_withincondi = pd.concat([temp_cur, final_temp_withincondi], axis=0)

    # Plotting stars
    if df_res.pval_1.to_numpy()[0] < bonfero_thresh and df_res.pval_2.to_numpy()[0] < bonfero_thresh:
        row_ind = [i for i in range(len(inner_name_list)) if incon == inner_name_list[i]][0]
        ax_ind0, ax_ind1 = get_ax_for_across_condi(compare_name, outer_name_list, anom, stat_type)
        
        if estimator_type == 'sum':
            across_marg0 = np.max(vec0)*(1/10)
            across_marg1 = np.max(vec1)*(1/10)
            ax[po, ax_ind0].text(row_ind-rshift, np.sum(vec0)-across_marg0, "*", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.sum(vec1)-across_marg1, "*", fontsize=fsize)
        elif estimator_type == 'mean':
            
            across_marg0 = np.max(vec0)*(1/10)
            across_marg1 = np.max(vec1)*(1/10)
            # ax[po, ax_ind0].text(row_ind-rshift, np.max(vec0)+across_marg0, "*", fontsize=fsize)
            # ax[po, ax_ind1].text(row_ind-rshift, np.max(vec1)+across_marg1, "*", fontsize=fsize)
            ax[po, ax_ind0].text(row_ind-rshift, np.max(vec0)-across_marg0, "*", fontsize=fsize)
            ax[po, ax_ind1].text(row_ind-rshift, np.max(vec1)-across_marg1, "*", fontsize=fsize)
            
    return final_temp_withincondi