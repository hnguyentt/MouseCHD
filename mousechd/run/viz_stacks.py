import os
import datetime
import imageio.v2 as imageio
import SimpleITK as sitk
import numpy as np
from mousechd.datasets.utils import get_bbx, get_largest_connectivity
from mousechd.utils.analyzer import (stack_slices, 
                                     stack_slices_with_ctn,
                                     stack_color_slices,
                                     plot_stacked_im,
                                     plot_stacked_ims)

def add_args(parser):
    parser.add_argument('-impath', type=str, help="Path to image")
    parser.add_argument('-crop', choices=["none", "axial", "heart"], help="Crop or not?", default="none")
    parser.add_argument('-annotate', type=int, choices=[0,1], help="Annotate heart with red box", default=0)
    parser.add_argument('-heart_cnt', type=int, choices=[0,1], help='draw heart contour (1), by default: 0', default=0)
    parser.add_argument('-maskpath', type=str, help='Path to mask', default=None)
    parser.add_argument('-pad', help='Padding values. Default: (0,0,0)', default="(0,0,0)")
    parser.add_argument('-num', type=int, help='Number of slices to stack', default=5)
    parser.add_argument('-trans_val', type=int, help='Translation value', default=10)
    parser.add_argument('-start_idx', type=int, 
                        help='Start index. Default: None (start from beginning, middle, and end)', 
                        default=None)
    parser.add_argument('-linewidth', type=float, help="linewidth", default=3)
    parser.add_argument('-bbx_color', help='bbx color', default='r')
    parser.add_argument('-plot_zline', type=int, choices=[0,1], help='Plot zline or not?', default=0)
    parser.add_argument('-zline_color', help='zline color', default='b')
    parser.add_argument('-plot_border', type=int, choices=[0,1], help='Plot border or not?', default=0)
    parser.add_argument('-color', help='border color or list of border colors seperated by ";"', default='b')
    parser.add_argument('-linestyle', help='linestyle for zline in multiple stack', default='--')
    parser.add_argument('-savedir', type=str, help='Directory to save image')
    
    
def main(args):
    pad = eval(args.pad)
    color = args.color.split(";")
    if len(color) == 1:
        color = color[0]
        
    today = datetime.datetime.now().strftime('%Y%m%d')
    os.makedirs(args.savedir, exist_ok=True)
    imname = args.impath.split(os.sep)[-1].split('.')[0]
    
    if bool(args.crop):
        assert args.maskpath is not None, "-maskpath is required for cropping"
        
    try:
        img = sitk.ReadImage(args.impath + "_0000")
    except RuntimeError:
        img = sitk.ReadImage(args.impath)
    im = sitk.GetArrayFromImage(img)
    mask = sitk.ReadImage(args.maskpath)
    ma = sitk.GetArrayFromImage(mask)
    # Post process
    ma[ma != 0] = 1
    max_clump = get_largest_connectivity(ma)
    ma[max_clump==0] = 0
    
    if args.crop:
            
        (min_z, max_z), (min_y, max_y), (min_x, max_x) = get_bbx(ma, pad=pad)
        
        if args.crop == "axial":
            im = im[min_z:max_z, :, :]
            ma = ma[min_z:max_z, :, :]
        else:
            im = im[min_z:max_z, min_y:max_y, min_x:max_x]
            ma = ma[min_z:max_z, min_y:max_y, min_x:max_x]
        
    if args.start_idx is None:
        start_idx_ls = [pad[0], im.shape[0]//2, im.shape[0] - pad[0] - args.num - 1]
    else:
        start_idx_ls = [args.start_idx, im.shape[0]//2, im.shape[0] - pad[0] - args.num - 1]
        
    outname_ls = []
    
    if args.crop == "none":
        suffix = ""
    elif (bool(args.annotate) and args.crop == "axial"):
        suffix = "_{}_ann".format(args.crop)
    else:
        suffix = "_{}".format(args.crop)
        
    if args.plot_zline:
        suffix += '_zline'
    if args.plot_border:
        suffix += '_border'
            
    for start_idx in start_idx_ls:
        print('Image shape: {}; num: {}; trans_val: {}; start_idx: {}'.format(im.shape,
                                                                              args.num,
                                                                              args.trans_val,
                                                                              start_idx))
        if args.heart_cnt == 1:
            stacked_im = stack_slices_with_ctn(im=im, 
                                               ma=ma, 
                                               num=args.num, 
                                               trans_val=args.trans_val, 
                                               start_idx=start_idx)
        else:
            stacked_im = stack_slices(im=im,
                                      num=args.num,
                                      trans_val=args.trans_val,
                                      start_idx=start_idx)
        
        if args.heart_cnt == 0:
            outname = "{}_{}_{}-{}-{}{}.png".format(today, imname, args.num, args.trans_val, start_idx, suffix)
        else:
            outname = "{}_{}_contour_{}-{}-{}{}.png".format(today, imname, args.num, args.trans_val, start_idx, suffix)
        
        outname_ls.append(outname)
        
        if (bool(args.annotate) and args.crop == "axial"):
            bbx = ((min_x, max_x), (min_y, max_y))
        else:
            bbx = None
        
        plot_stacked_im(stacked_im=stacked_im,
                        num=args.num,
                        trans_val=args.trans_val,
                        linewidth=args.linewidth,
                        bbx=bbx,
                        bbx_color=args.bbx_color,
                        plot_zline=bool(args.plot_zline),
                        zline_color=args.zline_color,
                        plot_border=bool(args.plot_border),
                        color=color,
                        save=os.path.join(args.savedir, outname)
                        )
        
    if len(outname_ls) > 1:
        print('Combining stacked_im')
        if args.heart_cnt == 0:
            multistack_name = "{}_{}_{}-{}{}_multistack.png".format(today, imname, args.num, args.trans_val, suffix)
        else:
            multistack_name = "{}_{}_contour_{}-{}{}_multistack.png".format(today, imname, args.num, args.trans_val, suffix)
        
        ims = []
        for outname in outname_ls:
            x = imageio.imread(os.path.join(args.savedir, outname))
            ims.append(x[::-1, :, :])
            
        ims = np.array(ims)
        
        stacked_ims = stack_color_slices(ims, len(outname_ls), max(ims.shape[1:])//2, 0)
        
        plot_stacked_ims(stacked_ims=stacked_ims,
                         slice_num=args.num,
                         slice_trans_val=args.trans_val,
                         num=len(outname_ls),
                         trans_val=max(ims.shape[1:])//2,
                         linewidth=args.linewidth,
                         color=args.zline_color,
                         linestyle=args.linestyle,
                         save=os.path.join(args.savedir, multistack_name)
                         )
        
        