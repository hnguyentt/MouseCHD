import logging
import tifffile
import os
import numpy as np
import skimage
import datetime

from mousechd.utils.tools import set_logger
from mousechd.utils.analyzer import (load_cams,
                                     create_gradcam_grid,
                                     plot_gallery)


def add_args(parser):
    parser.add_argument("-imdir", help="gradcam directory")
    parser.add_argument("-savedir", help="save directory")
    parser.add_argument("-hearts", help="list of hearts to visualize, separated by ';'")
    
    return parser


def main(args):
    today = datetime.datetime.now().strftime('%Y%m%d')
    os.makedirs(args.savedir, exist_ok=True)
    set_logger(os.path.join(args.savedir, "gradcam.log"))
    imls = args.hearts.split(";")
    logging.info("Parameters:")
    logging.info("imdir: {}".format(args.imdir))
    logging.info("savedir: {}".format(args.savedir))
    logging.info("hearts: {}".format(imls))
    suffix = ""
    for imname in imls:
        suffix += "-" + imname
    
    # 2D visualization
    ims, cams = load_cams(imdir=args.imdir, imls=imls)
    # for i, imname in enumerate(imls):
    #     for view in views:
    #         logging.info(f"Plot {imname} {view}")
    #         im = get_viewim(ims[i], view=view)
    #         cam = get_viewim(cams[i], view=view)
    #         fig = plot_gallery(ims=im, masks=cam, ncols=9, alpha=0.5)
    #         fig.savefig(os.path.join(args.savedir, f"{today}_CAM_{imname}_{view}.svg"),
    #                     bbox_inches="tight",
    #                     dpi=100)
        
    imgs, masks = create_gradcam_grid(ims, cams)
    logging.info("Plot GradCAM 2D")
    fig = plot_gallery(imgs, masks, ncols=3, alpha=0.5, figsize=(15,25))
    fig.savefig(os.path.join(args.savedir, f"{today}_gradcam2d{suffix}.svg"),
                bbox_inches="tight",
                dpi=100)
    
    # 3D visualization
    logging.info("Plot gradcam3d")
    import napari
    viewer = napari.Viewer()

    trans_val = 300
    for i, imname in enumerate(imls): 
        im = ims[i]
        cam = cams[i]
        # cam = skimage.color.rgb2gray(cam)
        
        viewer.add_image(im,
                        name=imname,
                        translate=[0, trans_val*i, 0],)
        viewer.add_image(cam,
                        name=f'cam_{imname}',
                        translate=[0, trans_val*i, 0],
                        colormap='jet',
                        contrast_limits=(0,250),
                        opacity=0.5)
        
    viewer.theme = 'system'
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, 45, 0)
    viewer.camera.zoom = 2

    
    viewer.screenshot(os.path.join(args.savedir, f"{today}_gradcam3d{suffix}.png"))

    napari.run()