import pywt
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from scipy.interpolate import interp1d
from mlxtend.preprocessing import minmax_scaling

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd

def scale_feature_data(feat, plotORnot):
    
    columns = ['0']
    dat = pd.DataFrame(data=feat, columns=columns)
    scaled_data0 = minmax_scaling(dat, columns=columns)
    scaled_data_mlx = list(scaled_data0.to_numpy().ravel())
    # OR 
    scaled_data_norma = []
    for q in range(len(feat)):
        scaled_data_norma.append( (feat[q] - np.min(feat))/(np.max(feat) - np.min(feat)) )  # normalization : same as mlxtend
    # OR 
    shift_up = [i - np.min(feat) for i in feat]
    scaled_data_posnorma = [q/np.max(shift_up) for q in shift_up]  # positive normalization : same as mlxtend
    # OR 
    scaled_data_standardization = [(q - np.mean(feat))/np.std(feat) for q in feat]  # standardization
    
    if plotORnot == 1:
        fig = make_subplots(rows=2, cols=1)
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        xxORG = list(range(len(feat)))
        fig.add_trace(go.Scatter(x=xxORG, y=feat, name='feat', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_mlx, name='scaled : mlxtend', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_norma, name='scaled : normalization', line = dict(color='cyan', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_posnorma, name='scaled : positive normalization', line = dict(color='blue', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_standardization, name='scaled : standardization', line = dict(color='orange', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.update_layout(title='feature vs scaled featue', xaxis_title='data points', yaxis_title='amplitude')
        fig.show(config=config)

    return scaled_data_mlx
    
    

def linear_intercurrentpt_makeshortSIGlong_interp1d(shortSIG, longSIG):

    x = np.linspace(shortSIG[0], len(shortSIG), num=len(shortSIG), endpoint=True)
    y = shortSIG
    # print('x : ', x)


    # -------------
    kind = 'linear'
    # kind : Specifies the kind of interpolation as a string or as an integer specifying the order of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

    if kind == 'linear':
        f = interp1d(x, y)
    elif kind == 'cubic':
        f = interp1d(x, y, kind='cubic')
    # -------------

    xnew = np.linspace(shortSIG[0], len(shortSIG), num=len(longSIG), endpoint=True)
    # print('xnew : ', xnew)

    siglong = f(xnew)

    return siglong



def tsig_2_continuous_wavelet_transform(t_sig, sig, scales, waveletname, plotORnot):
    
    dt = t_sig[1] - t_sig[0]
    
    [coefficients, frequencies] = pywt.cwt(sig, scales, waveletname, dt)
    
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    cmap = plt.cm.seismic
    
    x = t_sig 
    y = np.log2(period)
    z = np.log2(power)
    im = ax.contourf(x, y, z, contourlevels, extend='both',cmap=cmap) # matplotlib.contour.QuadContourSet object
    
    # ax.set_title('Wavelet Transform (Power Spectrum) of signal', fontsize=20)
    # ax.set_ylabel('Period', fontsize=18)
    # ax.set_xlabel('Time', fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    
    ylim = ax.get_ylim()
    # print('ylim: ', ylim)
    ax.set_ylim(ylim[0], -1)
    
    # Colorbar
    # cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    # fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    
    ax.axis('off')
    
    # -----------------------------------
    
    my_image = 'temp.png'
    fig.savefig(my_image)
    fname = os.path.abspath(os.getcwd()) + "/" +  my_image

    # Convert image to an array:
    # Read image 
    img = Image.open(fname)  

    # Convert image to an array:
    # Read image 
    img = Image.open(fname)         # PIL: img is not in array form, it is a PIL.PngImagePlugin.PngImageFile 

    # Plot raw pixel data
    # plt.imshow(img)
    
    # -----------------------------------
    
    # Flatten im
    dlen = len(sig)
    # print('dlen: ', dlen)

    num_px = int(np.floor(np.sqrt(dlen)))
    # print('num_px: ', num_px)

    rgb_image = img.convert('RGB')

    # Resize image into a 64, 64, 3
    new_h, new_w = int(num_px), int(num_px)
    img3 = rgb_image.resize((new_w, new_h), Image.ANTIALIAS)
    w_resized, h_resized = img3.size[0], img3.size[1]

    # Convert image to an array
    image = np.array(img3)

    type_out = '2D'
    # Transformer l'image à 2D ou 3D
    if type_out == '2D':  # get array back
        # Convert image back to a 2D array
        mat_resized = np.mean(image, axis=2)
    elif type_out == '3D':  # get image back
        mat_resized = image

    # Normalize the images to [-1, 1]
    mat_resized = (mat_resized - 127.5) / 127.5

    # plot raw pixel data
    # plt.imshow(mat_resized)

    # -----------------------------------

    # Convert image to an array
    image = np.array(mat_resized)
    # print('image.shape: ', image.shape)

    # Flatten image into a vector
    myimage_flatten = np.reshape(np.ravel(image), (num_px*num_px, ), order='F')
    # print('myimage_flatten.shape: ', myimage_flatten.shape)

    # print('taille de sig : ', len(sig))

    # Assurez que l'image taille est le meme que sig : interpolate
    img_flatten0 = linear_intercurrentpt_makeshortSIGlong_interp1d(myimage_flatten, sig)
    # print('img_flatten.shape: ', img_flatten.shape)

    # -----------------------------------
    
    # Normalize img_flatten values: on ne veut pas que des valeurs sont pres de zero
    img_flatten1 = scale_feature_data(img_flatten0, plotORnot=plotORnot)
    
    # -----------------------------------
    
    return img_flatten1
