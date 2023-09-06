import pandas as pd

def get_df(ax_val, ss_val, exp, df_timeseries_exp):
    
    if ax_val == 'all' and ss_val == 'all':
        # All the data
        df_timeseries_exp[exp].head()
        df = df_timeseries_exp[exp]
    elif ax_val != 'all' and ss_val == 'all':
        # Prediction for each axis
        if ax_val == 'ax0':
            ax_val_n = 0
        elif ax_val == 'ax1':
            ax_val_n = 1
        elif ax_val == 'ax2':
            ax_val_n = 2
        df = df_timeseries_exp[exp][(df_timeseries_exp[exp].ax == ax_val_n)]
    elif ax_val == 'all' and ss_val != 'all':
        # Prediction per sup/sub
        if ss_val == 'sup':  # sup
            df = df_timeseries_exp[exp][(df_timeseries_exp[exp].ss > 0)]
        elif ss_val == 'sub':  # sub
            df = df_timeseries_exp[exp][(df_timeseries_exp[exp].ss < 0)]
    elif ax_val != 'all' and ss_val != 'all':
        # Prediction per axis and sup/sub
        if ax_val == 'ax0':
            ax_val_n = 0
        elif ax_val == 'ax1':
            ax_val_n = 1
        elif ax_val == 'ax2':
            ax_val_n = 2
        
        if ss_val == 'sup':  # sup
            df = df_timeseries_exp[exp][(df_timeseries_exp[exp].ss > 0) & (df_timeseries_exp[exp].ax == ax_val_n)]
        elif ss_val == 'sub':  # sub
            df = df_timeseries_exp[exp][(df_timeseries_exp[exp].ss < 0) & (df_timeseries_exp[exp].ax == ax_val_n)]

    print('Confirmation : exp=', exp, ', ax_val=', ax_val, ', ss_val=', ss_val)
    
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    
    return df
