"""
Comparison of manual and automatic segmentation
"""
import SimpleITK as sitk
import os
import datetime
import argparse

from mousechd.datasets.utils import anyview2LPS, get_largest_connectivity
from mousechd.datasets.utils import make_isotropic

def load_img(impath, spacing=None):
    img = sitk.ReadImage(impath)
    img = anyview2LPS(img)
    img = make_isotropic(img, spacing=spacing)
    
    return sitk.GetArrayFromImage(img)

def add_args(parser):
    parser.add_argument('-imdir', type=str, help='image directory')
    parser.add_argument('-maskdir', type=str, help='manual mask directory')
    parser.add_argument('-preddir', type=str, help='predicted mask directory')
    parser.add_argument('-imname', type=str, help='image name (without extension, image must be in .nii.gz format)')
    parser.add_argument('-savedir', type=str, help='save directory', default=None)
    parser.add_argument('-kwargs', type=str, help='a dictionary of for extra options', default="{}")
    
def main(args):
    import napari
    
    today = datetime.datetime.now().strftime('%Y%m%d')
    kwargs = eval(args.kwargs)
    cmap = kwargs.get('cmap', 'jet')
    try:
        cmap = napari.utils.colormaps.ensure_colormap(cmap)
    except KeyError:
        print(f'{cmap} is not available, use cmap "jet" instead')
        cmap = napari.utils.colormaps.ensure_colormap('jet')
    
    img = load_img(os.path.join(args.imdir, args.imname + "_0000"))
    mask = load_img(os.path.join(args.maskdir, args.imname))
    pred = load_img(os.path.join(args.preddir, args.imname))
    pred = get_largest_connectivity(pred)
    mask[mask !=0 ] = 1
    pred[pred != 0] = 2
    merged = mask + pred
    
    viewer = napari.Viewer()
    viewer.add_image(img,
                     name=args.imname)
    viewer.add_image(merged,
                     name='mask_{}'.format(args.imname),
                     colormap=cmap,
                     opacity=kwargs.get('opacity', 0.5)
                     )
    
    viewer.theme = 'light'
    viewer.dims.ndisplay = 3
    viewer.camera.angles = kwargs.get('angles', (0,45,0))
    
    if args.savedir is not None:
        os.makedirs(args.savedir, exist_ok=True)
        fname = '{}_{}_seg3d.png'.format(today, args.imname)
        viewer.screenshot(os.path.join(args.savedir, fname))
        print('Screenshot saved as {}'.format(os.path.join(args.savedir, fname)))
    
    napari.run()
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser('Visualize 3D segmentation')
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
    