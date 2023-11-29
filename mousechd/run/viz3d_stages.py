import os
import datetime
import time
import argparse
import numpy as np
import imageio.v2 as imageio
import SimpleITK as sitk
from skimage.transform import rescale
from mousechd.datasets.utils import (crop2dimage, 
                                     get_bbx,
                                     get_largest_connectivity)
from mousechd.utils.analyzer import (gen_white2blue_cmap,
                                     gen_white2red_cmap,
                                     build_3d_bbx)

def add_args(parser):
    parser.add_argument('-imdir', type=str, help='Image directory')
    parser.add_argument('-maskdir', help='Mask directory', default=None)
    parser.add_argument('-stage', help='Stage')
    parser.add_argument('-imname', help='Image name')
    parser.add_argument('-savedir', help='Save image directory', default=None)
    parser.add_argument('-suffix', help='Filename suffix', default=None)
    parser.add_argument('-rescale_ratio', help='Rescale ratio, default: 0.5', default=0.5)
    parser.add_argument('-bbx', choices=["none", "heart", "axial", "both"], help='Display bounding box around the heart', default="none")
    parser.add_argument('-pad', help='Padding when cropping bbx', default="(1,1,1)")
    parser.add_argument('-zoom', type=float, help='Camera zoom factor', default=None)
    parser.add_argument('-cam_angles', help='Camera angles', default="(0,45,0)")
    parser.add_argument('-crop', help='Crop image', type=int, choices=[0,1], default=0)
    parser.add_argument('-left_off', help='Ratio to cut from the left, default: 0', type=float, default=0)
    parser.add_argument('-right_off', help='Ratio to cut from the right, default: 0', type=float, default=0)
    parser.add_argument('-top_off', help='Ratio to cut from the right, default: 0', type=float, default=0)
    parser.add_argument('-bottom_off', help='Ratio to cut from the bottom, default: 0', type=float, default=0)
    parser.add_argument('-im_opac', help='Image opacity', type=float, default=1)
    parser.add_argument('-heart_cmap', help="Heart cmap", default="turbo")
    parser.add_argument('-heart_opac', help='Heart opacity', type=float, default=0.75)
    parser.add_argument('-bbxaxial_opac', help='Opacity of bbxaxial', type=float, default=0.25)
    parser.add_argument('-bbxheart_opac', help='Opacity of bbxheart', type=float, default=0.3)
    
    
def main(args):
    import napari
    from napari.utils.notifications import show_info

    assert (args.left_off + args.right_off) < 1, 'left_off + right_off must be < 1'
    assert (args.top_off + args.bottom_off) < 1, 'top_off + bottom_off must be < 1'
    
    imdir = args.imdir
    maskdir = args.maskdir
    stage = args.stage
    imname = args.imname
    savedir = args.savedir
    today = datetime.datetime.now().strftime('%Y%m%d')
    if args.suffix is not None:
        filename = "{}_{}_{}_{}".format(today, stage, imname, args.suffix)
    else:
        filename = "{}_{}_{}".format(today, stage, imname)
    
    viewer = napari.Viewer()
    show_info('Loading images. This process may take ~ 1 minute. Be patient!')
    
    start_time = time.time()
    
    img = sitk.ReadImage(os.path.join(imdir, imname + "_0000"))
    ori_im = sitk.GetArrayFromImage(img)
    im = rescale(ori_im, args.rescale_ratio)
    im = (im - im.min()) / (im.max() - im.min())
    print('Finish loading and processing image. Current processing time {}'.format(
        time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-start_time))
    ))
    viewer.add_image(im,
                     name=stage,
                     colormap='twilight',
                     gamma=2.,
                     opacity=args.im_opac,
                     contrast_limits=(0.158, 0.946),
                    #  contrast_limits=(0., 0.974),
                     )
    
    # Mask
    if maskdir is not None:
        mask = sitk.ReadImage(os.path.join(maskdir, imname))
        ma = sitk.GetArrayFromImage(mask)
        bin_mask = ma.copy()
        bin_mask[bin_mask != 0] = 1
        max_clump = get_largest_connectivity(bin_mask)
        ma = ori_im.copy()
        ma[max_clump==0] = 0
        ma = rescale(ma, args.rescale_ratio)
        ma = (ma - ma.min()) / (ma.max() - ma.min())
        print('Finish loading and processing mask. Current processing time {}'.format(
            time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-start_time))
            ))
        
        viewer.add_image(ma,
                         name=f"mask_{stage}",
                         colormap=args.heart_cmap,
                         opacity=args.heart_opac,
                         contrast_limits=(0.520, 0.791),
                        #  contrast_limits=(0.1, 0.8)
                         )
        
        if args.bbx != "none":
            pad = eval(args.pad)
            (min_z, max_z), (min_y, max_y), (min_x, max_x) = get_bbx(ma, pad=pad)
            
            if args.bbx in ['axial', 'both']:
                trans_vals = (min_z, 0, 0)
                bbx_shape = (max_z-min_z, im.shape[1], im.shape[0])
                bbx = build_3d_bbx(bbx_shape, 1)
                viewer.add_image(bbx,
                                 name=f"axial_{stage}",
                                 colormap=gen_white2blue_cmap(),
                                 opacity=args.bbxaxial_opac,
                                 contrast_limits=(-0.5,1),
                                 translate=trans_vals)
                
            if args.bbx in ['heart', 'both']:
                trans_vals = (min_z, min_y, min_x)
                bbx_shape = (max_z-min_z, max_y-min_y, max_x-min_x)
                bbx = build_3d_bbx(bbx_shape, 1)
                viewer.add_image(bbx,
                                 name=f"heart_{stage}",
                                 colormap=gen_white2red_cmap(),
                                 opacity=args.bbxheart_opac,
                                 contrast_limits=(-0.5,1),
                                 translate=trans_vals)
        
    viewer.theme = 'light'
    viewer.dims.ndisplay = 3
    if args.zoom is not None:
        viewer.camera.zoom = args.zoom
    viewer.camera.angles = eval(args.cam_angles)
    
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        
        viewer.screenshot(os.path.join(savedir, f"{filename}.png"))
        print('Screenshot saved as {}'.format(os.path.join(savedir, f"{filename}.png")))
        
        if bool(args.crop):
            im = imageio.imread(os.path.join(savedir, f"{filename}.png"))
            im = crop2dimage(im=im,
                            left_off=args.left_off,
                            right_off=args.right_off,
                            top_off=args.top_off,
                            bottom_off=args.bottom_off)
            imageio.imwrite(os.path.join(savedir, f"{filename}_cropped.png"), im)
            print('Cropped image saved as {}'.format(
                os.path.join(savedir, f"{filename}_cropped.png")))
            
            print('Finish cropping image. Current processing time {}'.format(
                time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-start_time))
            ))
    
    print('Total processing time {}'.format(time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-start_time))))
        
    napari.run()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='VISUALIZE DIFFERENT STAGES')
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
        