"""
Utilities for visualization
"""
import logging
import io
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from PIL import Image
import tifffile
from venn import venn
import matplotlib.pyplot as plt
mpl.use('Agg')

from mousechd.datasets.utils import get_bbx

EXCLUDE_IDS = ["N_261h", "NH_229m"]
views = ["axial", "coronal", "sagittal"]

#########
# Utils #
#########

def load_metadata(meta_path, sep=","):
    """
    Load metadata
    
    Args:
        + meta_path (str): path to metadata file
        + sep (str): delimiter of csv file
     
    Return:
        + (pd.DataFrame): metadata dataframe
    """
    df = pd.read_csv(meta_path, sep=sep)
    df = df[df.Stage.isin(['P0', 'E18.5', 'E17.5'])] # Get only stages: P0, E18.5, E17.5
    df = df[~df.heart_name.isin(EXCLUDE_IDS)]
    df.set_index('heart_name', inplace=True)
    
    return df


def get_kingdom_df(terms, df):
    """
    Get kingdom dataframe
    
    Args:
        + terms (str): dataframe of terminology
        + df (str): dataframe of metadata
    
    Return:
        + (pd.DataFrame): kingdom dataframe
    """
    trans_df = df.drop('Stage', axis=1).transpose()
    trans_df['Kingdom'] = trans_df.index.map(terms.set_index('Diagnostic').to_dict()['Kingdom'])
    kingdom_df = trans_df.groupby('Kingdom').agg('sum').reset_index()
    kingdom_df = kingdom_df[kingdom_df[kingdom_df.columns[1:]].sum(axis=1) > 0]
    kingdom_df.set_index('Kingdom', inplace=True)
    kingdom_df = kingdom_df.transpose()
    
    return kingdom_df


def set_default_mplstyle(**kwargs):
    "Set default style for matplotlib plot"
    # Default params
    if kwargs is None:
        kwargs = {}
        
    params = {
        'figure.titlesize': 20,
        'figure.figsize': (7., 4.),
        'figure.dpi': 200,
        'font.family': 'Arial',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 3.,
        'axes.linewidth': 1.5,
        'axes.labelsize': 15,
        'legend.fontsize': 12,
        'legend.title_fontsize': 13,
        'axes.grid': False,
        'image.cmap': 'Accent'
    }
    
    for k, v in params.items():
        mpl.rcParams[k] = kwargs.get(k, v)


def plot2image(figure):
    """Converts a matplotlib plot specified by `figure` to a PNG image and returns it.
    The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    # Convert PNG buffer to PIL image
    buf.seek(0)
    
    im = Image.open(buf)
    
    return im


###########
# Figures #
###########
def plot_kingdom_venn(kingdom_df, save=None, **kwargs):
    """
    Plot venn diagram based on Kingdom
    """
    set_default_mplstyle(**kwargs)
    
    diseases = list(kingdom_df.columns)
    try:
        diseases.remove('Normal heart')
    except ValueError:
        pass
    
    subsets = {}
    hearts = []

    for d in diseases:
        subsets[d] = set(kingdom_df[kingdom_df[d]>0].index)
        hearts += list(kingdom_df[kingdom_df[d]>0].index)
        
    venn(subsets,
         fontsize=kwargs.get('fontsize', 12),
         figsize=kwargs.get('figsize', (3.5,3.5)),
         legend_loc=kwargs.get('legend_loc', (1, 0.25))
        );

    plt.text(0.15, 0.8,'N={}'.format(len(set(hearts))), 
            fontsize=kwargs.get('fontsize', 12), 
            weight='bold',
            ha='center');
    
    if save is not None:
        # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig(save,
                    pad_inches=0, 
                    bbox_inches='tight', 
                    transparent=True, 
                    dpi=kwargs.get('dpi', 200))


def stack_slices(im, num, trans_val, start_idx):
    """
    Stack a number of grayscale slices along first axis, starting with an indicated index
    
    Args:
        im (np.array): 3D-array of image
        num (int): number of slices to stack
        trans_val (int): number of pixels for translation every each stacked slice
        start_idx (int): starting index
    
    Returns:
        (np.array): 2D array of stacked slices (W x H)
    """
    stacked_im = np.zeros((im.shape[1]+(num-1)*trans_val, im.shape[2]+(num-1)*trans_val)) - 1 # -1 for masking
    stacked_im[:im.shape[1], :im.shape[2]] =  im[start_idx, :, :]
    
    for i in range(1, num):
        x = im[start_idx+i, :, :]
        stacked_im[(x.shape[0]+(i-1)*trans_val):(x.shape[0]+i*trans_val), 
                   i*trans_val:(x.shape[1]+i*trans_val)] = x[-trans_val:, :]
        stacked_im[i*trans_val:(x.shape[0]+i*trans_val), 
                   (x.shape[1]+(i-1)*trans_val):(x.shape[1]+i*trans_val)] = x [:, -trans_val:]
        
    stacked_im = np.ma.masked_where(stacked_im==-1, stacked_im)
    
    return stacked_im


def stack_slices_with_ctn(im, ma, num, trans_val, start_idx):
    """Stack a number of slices with contours along first axis, starting with an indicated index

    Args:
        im (np.array): 3D-array of image
        ma (np.array): 3D-array of mask
        num (int): number of slices to stack
        trans_val (int): number of pixels for translation every each stacked slice
        start_idx (int): starting index
        
    Returns:
        (np.array): 4D array of RGBA of stacked slices (W x H x C x A) [C: # of channels, A: alpha values]
    """
    from skimage import segmentation
    epsilon = 10e-5
    
    stacked_im = np.zeros((im.shape[1]+(num-1)*trans_val, im.shape[2]+(num-1)*trans_val, 3))
    
    # Add contour
    clean_border = segmentation.clear_border(ma[start_idx].astype(int))
    edges = segmentation.mark_boundaries(im[start_idx], clean_border)
    im2d = np.stack([im[start_idx]]*3, axis=2)
    cnt_im = (im2d - im2d.min())/(im2d.max() - im2d.min()) + edges
    cnt_im = (cnt_im - cnt_im.min() + epsilon)/(cnt_im.max() - cnt_im.min() + epsilon)
    
    stacked_im[:im.shape[1], :im.shape[2], :] =  cnt_im
    
    for i in range(1, num):
        clean_border = segmentation.clear_border(ma[start_idx + i].astype(int))
        edges = segmentation.mark_boundaries(im[start_idx + i], clean_border)
        im2d = np.stack([im[start_idx + i]]*3, axis=2)
        im2d = (im2d - im2d.min())/(im2d.max() - im2d.min())
        x = im2d + edges
        x = (x - x.min() + epsilon)/(x.max() - x.min() + epsilon)
        
        stacked_im[(x.shape[0]+(i-1)*trans_val):(x.shape[0]+i*trans_val), 
                   i*trans_val:(x.shape[1]+i*trans_val), :] = x[-trans_val:, :, :]
        
        stacked_im[i*trans_val:(x.shape[0]+i*trans_val), 
                   (x.shape[1]+(i-1)*trans_val):(x.shape[1]+i*trans_val), :] = x [:, -trans_val:, :]
        
    alphas = np.ones((stacked_im.shape[0], stacked_im.shape[1]))
    merged = stacked_im.mean(axis=2)
    alphas[merged==0] = 0
    stacked_im = np.stack([stacked_im[:,:,0], stacked_im[:,:,1], stacked_im[:,:,2], alphas], axis=2)
    
    return stacked_im


def stack_color_slices(im, num, trans_val, start_idx):
    #TODO: Write discription
    stacked_im = np.zeros((im.shape[1]+(num-1)*trans_val, 
                           im.shape[2]+(num-1)*trans_val,
                           im.shape[3]))
    stacked_im[:im.shape[1], :im.shape[2], :] =  im[start_idx, :, :, :]
    
    for i in range(1, num):
        x = im[start_idx+i, :, :, :]
        stacked_im[(x.shape[0]+(i-1)*trans_val):(x.shape[0]+i*trans_val), 
                   i*trans_val:(x.shape[1]+i*trans_val), :] = x[-trans_val:, :, :]
        stacked_im[i*trans_val:(x.shape[0]+i*trans_val), 
                   (x.shape[1]+(i-1)*trans_val):(x.shape[1]+i*trans_val), :] = x[:, -trans_val:, :]
        
    stacked_im = np.ma.masked_where(stacked_im==0, stacked_im)
    
    return stacked_im.astype("uint8")


def plot_stacked_im(stacked_im, **kwargs):
    """
    Plot stacked image from function stack_slices or stack_color_slices
    
    Args:
        stacked_im (np.array): result from function stack_slices
        kwargs:
            + slice_shape (tuple-like): shape of slice in stacked_im (H x W)
            + num (int): number of slices in stacked_im
            + trans_val (int): translation value
            + linewidth (float): linewidth for border
            + bbx (array-like): bounding box ((min_x, max_x), (min_y, max_y)) or None
            + bbx_color (str): color code for bounding box
            + plot_zline (bool): if plot zline
            + zline_color (str): color code for zline
            + plot_border (bool): if plot border
            + color (str|list): color for each slice
            + save (str): path to save (default: None) 
    Return:
        + (plt.Figure): figure
    """
    num = kwargs.get('num', None)
    trans_val = kwargs.get('trans_val', None)
    linewidth = kwargs.get('linewidth', 3)
    bbx = kwargs.get('bbx', None)
    bbx_color = kwargs.get('bbx_color', 'r')
    plot_zline = kwargs.get('plot_zline', False)
    zline_color = kwargs.get('zline_color', 'b')
    plot_border = kwargs.get('plot_border', False)
    color = kwargs.get('color', 'b')
    save = kwargs.get('save', None)
    
    if plot_border or plot_zline:
        logging.info("'num', and 'trans_val' (number of stacked slices) are required to plot_border and plot_zline." + 
                     "If not specified, border and zline will not be plotted!")
        
        if (num is None) or (trans_val is None):
            plot_border = False
            plot_zline = False
        else:
            slice_shape = (int(stacked_im.shape[0]-(num-1)*trans_val), int(stacked_im.shape[1]-(num-1)*trans_val))
        
    if plot_border and (slice_shape is not None) and (type(color) == str):
        color = [color] * num
        assert len(color) >= num, 'Expected at least {} colors but get {}'.format(num, len(color))
        
    fig, ax = plt.subplots(figsize=(10, stacked_im.shape[0]*10/stacked_im.shape[1]))
    ax.imshow(stacked_im, cmap='gray', origin='lower')
    
    if bbx is not None:
        (min_x, max_x), (min_y, max_y) = bbx
        ax.add_patch(Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                               linewidth=linewidth, 
                               edgecolor=bbx_color,
                               facecolor='none'))
        
    if plot_border:
        ax.add_patch(Rectangle((0, 0), slice_shape[1], slice_shape[0],
                               linewidth=linewidth,
                               edgecolor=color[0],
                               facecolor='none'))
        for i in range(1, num):
            ax.plot([i*trans_val, i*trans_val+slice_shape[1]-1, slice_shape[1]+i*trans_val-1],
                    [i*trans_val+slice_shape[0]-1, i*trans_val+slice_shape[0]-1, i*trans_val],
                    c=color[i],
                    linewidth=linewidth)
            
            if plot_zline:
                    linestyle = '--'
            else:
                    linestyle = '-'
            ax.plot([i*trans_val, i*trans_val],
                    [slice_shape[0]+(i-1)*trans_val, slice_shape[0]+i*trans_val-1],
                    c=color[i],
                    linestyle=linestyle,
                    linewidth=linewidth)
            ax.plot([slice_shape[1]+(i-1)*trans_val, slice_shape[1]+i*trans_val-1],
                    [i*trans_val, i*trans_val],
                    c=color[i],
                    linestyle=linestyle,
                    linewidth=linewidth)
            
    if plot_zline:
        ax.plot([slice_shape[1], stacked_im.shape[1]-1],
                [0, (num-1)*trans_val],
                c=zline_color,
                linestyle='-',
                linewidth=linewidth)
        ax.plot([slice_shape[1], stacked_im.shape[1]-1],
                [slice_shape[0], stacked_im.shape[0]-1],
                c=zline_color,
                linestyle='-',
                linewidth=linewidth)
        ax.plot([0, (num-1)*trans_val],
                [slice_shape[0], stacked_im.shape[0]-1],
                c=zline_color,
                linestyle='-',
                linewidth=linewidth)
    
    ax.axis("off")
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    
    if save is not None:
        fig.savefig(save, 
                    pad_inches=0, 
                    transparent=True,
                    bbox_inches='tight',
                    dpi=stacked_im.shape[0]/fig.get_size_inches()[1])
    
    return fig


def plot_stacked_ims(stacked_ims, 
                     slice_num, 
                     slice_trans_val,
                     num,
                     trans_val, 
                     **kwargs):
    """
    Plot stacked image from function stack_color_slices
    
    Args:
        + stacked_ims (np.array): output of stack_color_slices
        + slice_num (int): number of slices in stacked_ims
        #TODO: Continue description
    """
    linewidth = kwargs.get('linewidth', 3)
    color = kwargs.get('color', 'b')
    linestyle = kwargs.get('linestyle', '--')
    im_shape = (int(stacked_ims.shape[0]-(num-1)*trans_val), int(stacked_ims.shape[1]-(num-1)*trans_val))
    save = kwargs.get('save', None)
    
    fig, ax = plt.subplots(figsize=(10, stacked_ims.shape[0]*10/stacked_ims.shape[1]))
    ax.imshow(stacked_ims, cmap='gray', origin='lower')
    
    im_pad = (slice_num-1)*slice_trans_val
    delta = trans_val - im_pad
    for i in range(1, num):
        ax.plot([i*im_pad+(i-1)*delta, i*(im_pad+delta)],
                [im_shape[0]+(i-1)*(delta+im_pad), im_shape[0]+i*delta+(i-1)*im_pad],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth)
        ax.plot([im_shape[1]+(i-1)*(delta+im_pad), im_shape[1]+i*delta+(i-1)*im_pad],
                [i*im_pad+(i-1)*delta, i*(im_pad+delta)],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth)
        ax.plot([im_shape[1]+(i-1)*(delta+im_pad), im_shape[1]+i*delta+(i-1)*im_pad],
                [im_shape[0]+(i-1)*(delta+im_pad), im_shape[0]+i*delta+(i-1)*im_pad],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth)
    
    ax.axis("off")
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    
    if save is not None:
        fig.savefig(save, 
                    pad_inches=0, 
                    transparent=True,
                    bbox_inches='tight',
                    dpi=stacked_ims.shape[0]/fig.get_size_inches()[1])
    
    return fig


def plot_contingency(data, x, y, save=None, **kwargs):
    """
    Plot contingency
    
    Args:
        + data (pd.DataFrame): dataframe containing two columns of interested variables
        + x (str): name of variable for x axis
        + y (str): name of variable for y axis
        + save (str): path to file to save (default: None)
        + kwargs (dict):
            * fontsize (float): fontsize for text on contingency table
            * #TODO: write more description
    """
    fontsize = kwargs.get('fontsize', 12)
    ax_fs = fontsize*1.1
    font_scale = kwargs.get('fontscale', 1.5)
    figsize = kwargs.get('figsize', (2.75, 2.75))
    annot = kwargs.get('annot', True)
    annot_kws = kwargs.get('annot_kws', {'size': fontsize})
    fmt = kwargs.get('fmt', 'd')
    cmap = kwargs.get('cmap', "Blues")
    cbar = kwargs.get('cbar', False)
    cbar_kws = kwargs.get('cbar_kws', {})
    box_colors = kwargs.get('box_colors', ['orange', 'wheat'])
    if isinstance(box_colors, str):
        box_colors = [box_colors]*2
    box_alpha = kwargs.get('box_alpha', 0.7)
    
    tb = pd.crosstab(data[y], data[x])
    sns.set(font_scale=font_scale)
    plt.figure(figsize=figsize)
    g = sns.heatmap(tb,
                    annot=annot,
                    annot_kws=annot_kws,
                    fmt=fmt,
                    cmap=cmap,
                    cbar=cbar,
                    cbar_kws=cbar_kws)
    g.set_xlabel(g.get_xlabel(), fontsize=ax_fs)
    g.set_ylabel(g.get_ylabel(), fontsize=ax_fs)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=ax_fs)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=ax_fs)
    
    xs = tb.columns.to_list()
    ys = tb.index.to_list()
    x1_per_x2 = tb[xs[0]]/tb[xs[1]]
    y1_per_y2 = tb.loc[ys[0], :]/tb.loc[ys[1], :]

    props1 = dict(boxstyle='round', facecolor=box_colors[0], alpha=box_alpha)
    props2 = dict(boxstyle='round', facecolor=box_colors[1], alpha=box_alpha)
    
    # Add ratio boxes
    plt.text(1,0.2, 
             r'$\frac{{{}}}{{{}}}\approx$'.format(xs[0], xs[1]) + '{:.2f}'.format(x1_per_x2[ys[0]]),
             fontsize=fontsize, va='center', ha='center', bbox=props1)
    
    plt.text(1,1.8, 
             r'$\frac{{{}}}{{{}}}\approx$'.format(xs[0], xs[1]) + '{:.2f}'.format(x1_per_x2[ys[1]]),
             fontsize=fontsize, va='center', ha='center', bbox=props1)
    
    plt.text(0.5,1, 
             r'$\frac{{{}}}{{{}}}\approx$'.format(ys[0], ys[1]) + '{:.2f}'.format(y1_per_y2["CHD"]),
             fontsize=fontsize, va='center', ha='center', bbox=props2)
    
    plt.text(1.5,1, 
             r'$\frac{{{}}}{{{}}}\approx$'.format(ys[0], ys[1]) + '{:.2f}'.format(y1_per_y2["Normal"]),
             fontsize=fontsize, va='center', ha='center', bbox=props2);
    
    if save is not None:
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig(save,
                    pad_inches=0, 
                    bbox_inches='tight', 
                    transparent=True, 
                    dpi=kwargs.get('dpi', 200))
        

        
########################
# UTILITIES FOR NAPARI #
########################
def gen_white2blue_cmap(num=255):
    """
    Generate white to blue colormap with transparent at 0
    """
    colors = np.linspace(
        start=[1, 1, 1, 1],
        stop=[0, 0, 1, 1],
        num=num,
        endpoint=True
    )
    colors[0] = np.array([1., 1., 1., 0])
    new_colormap = {
    'colors': colors,
    'name': 'white2blue',
    'interpolation': 'linear'
    }
    
    return new_colormap

def gen_white2red_cmap(num=255):
    """
    Generate white to red colormap with transparent at 0
    """
    colors = np.linspace(
        start=[1, 1, 1, 1],
        stop=[1, 0, 0, 1],
        num=num,
        endpoint=True
    )
    colors[0] = np.array([1., 1., 1., 0])
    new_colormap = {
    'colors': colors,
    'name': 'white2red',
    'interpolation': 'linear'
    }
    
    return new_colormap


def build_3d_bbx(shape, b):
    """
    Build bounding box with line border to display on Napari
    
    Args:
        shape (tuple): 3-element tuple of shape for bounding box
        b (int): border thickness
    
    Returns:
        Bounding box (np.array)
    """
    min_val = 0
    max_val = 1
    x = np.zeros(shape)
    x[ :b, :, :] = max_val
    x[-b:, :, :] = max_val
    x[:,  :b, :] = max_val
    x[:, -b:, :] = max_val
    x[:, :,  :b] = max_val
    x[:, :, -b:] = max_val
    x[:b, b:-b, b:-b] = min_val
    x[-b:, b:-b, b:-b] = min_val
    x[b:-b, :b, b:-b] = min_val
    x[b:-b, -b:, b:-b] = min_val
    x[b:-b, b:-b, :b] = min_val
    x[b:-b, b:-b, -b:] = min_val
    
    return x


def plot_gallery(ims, masks=None, ncols=4, **kwargs):
    """
    Plot image gallery given a list of images and masks (optional)
    
    Args:
        ims (list): a list of 2D images
        masks (list): a list of 2D masks
        ncols (int): number of columns
    
    Returns:
        fig (plt.figure): figure of gallery
    """
    x_max = max([x.shape[0] for x in ims])
    y_max = max([x.shape[1] for x in ims])
    
    if masks is None:
        masks = [None] * len(ims)
    else:
        vmins = np.array([x.min() for x in masks if x is not None])
        vmaxs = np.array([x.max() for x in masks if x is not None])
        vmin = np.min(vmins)
        vmax = np.max(vmaxs)
        
    nrows = int(np.ceil(len(ims)/ncols))
    fig = plt.figure(figsize=kwargs.get('figsize', (ncols*3, nrows*3)))
    axes = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=kwargs.get('axes_pad', 0.1),
                     share_all=True)
    for i, im in enumerate(ims):
        axes[i].imshow(im, cmap=kwargs.get('cmap', 'gray'))
        if masks[i] is not None:
            axes[i].imshow(np.ma.masked_where(masks[i]==0, masks[i]), 
                           cmap=kwargs.get('mask_cmap', 'jet'), 
                           vmin=vmin, vmax=vmax, 
                           alpha=kwargs.get('alpha', 1.))
        
    for i in range(ncols*nrows):
        axes[i].set_xlim([0,x_max])
        axes[i].set_ylim([0,y_max])
        axes[i].axis('off')
            
    return fig


def load_cams(imdir, imls):
    ims = []
    cams = []
    for imname in imls:
        ims.append(tifffile.imread(os.path.join(imdir, imname, "image.tif")))
        cams.append(tifffile.imread(os.path.join(imdir, imname, "cam.tif")))
    
    return ims, cams


def get_viewim(ims, view="axial"):
    assert view in views, f"view must be in {views}"
    if view == "axial":
        return [ims[i,:,:] for i in range(ims.shape[0])]
    elif view == "coronal":
        return [ims[:,i,:] for i in range(ims.shape[1])]
    else:
        return [ims[:,:,i] for i in range(ims.shape[2])]
    

def create_gradcam_grid(ims, cams, idx=None):
    """Create image and cam grid to plot Figure for GradCAMs

    Args:
        ims (np.array): 3D image
        cams (np.array): 3D cams corresponding to ims

    Returns:
        tuple: reorganized imgs and maks
    """
    imgs = []
    masks = []
    if idx is None:
        idx = [im.shape[0]//2, im.shape[1]//2, im.shape[2]//2]
    else:
        assert len(idx) == 3, "idx must contain 3 integers"
        
    for im, cam in zip(ims, cams):
        imgs += [im[idx[0], :, :],
                im[:, idx[1], :],
                im[:, :, idx[2]]]
        imgs += [im[idx[0], :, :],
                im[:, idx[1], :],
                im[:, :, idx[2]]]
        masks += [None, None, None,
                cam[idx[0], :, :],
                cam[:, idx[1], :],
                cam[:, :, cam.shape[2]//2]]
        
    return imgs, masks


########################
# Analyze segmentation #
########################
def count_heart_slices(maskdir, hearts=None, df=None):
    if hearts is None:
        hearts = [re.sub(r".nii.gz$", "", x) for x in os.listdir(maskdir) 
                  if x.endswith('nii.gz') and not x.startswith('.')]
    
    logging.info("Total: {}".format(len(hearts)))
    if df is not None:
        processed = df["heart_name"].to_list()
        logging.info("Processed: {}".format(len(processed)))
        hearts = [x for x in hearts if x not in processed]
        logging.info("Need to process: {}".format(len(hearts)))
    
    for i, imname in enumerate(hearts):
        logging.info("{}. {}".format(i+1, imname))
        img = sitk.ReadImage(os.path.join(maskdir, imname))
        im = sitk.GetArrayFromImage(img)
        (min_z, max_z), (min_y, max_y), (min_x, max_x) = get_bbx(im)
        logging.info("=====> axial: {}, coronal: {}, sagittal: {}".format(max_z - min_z + 1,
                                                                          max_y - min_y + 1,
                                                                          max_x - min_x + 1))
        if df is not None:
            df.loc[len(df), :] = [imname,
                                  max_z - min_z + 1,
                                  max_y - min_y + 1,
                                  max_x - min_x + 1]
    
    return df
        
