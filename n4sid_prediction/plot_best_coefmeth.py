# Created by Jamilah Foucher, Aout 25, 2021 

# Purpose : Plots the input, output, and output prediction.

# Plotting
import plotly.graph_objects as go



def plot_best_coefmeth(time, inputs_org, outputs_org, pred_output, coef_meth_best, metric_val_best, plot_final_prediction):
    
    # --------------------
    # Plot the best predicted output
    # --------------------
    if plot_final_prediction == 1:
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        fig.add_trace(go.Scatter(x=time, y=inputs_org, name='inputs_org', line = dict(color='green', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=time, y=outputs_org, name='outputs_org', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=time, y=pred_output, name='predicted output', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        
        fig.update_layout(title='Final prediction : coef method=%d: metric val=%5.5f' % (coef_meth_best, metric_val_best), xaxis_title='time', yaxis_title='signal')
        fig.show(config=config)
        # fig.write_image("fig.png")
    # --------------------

    return