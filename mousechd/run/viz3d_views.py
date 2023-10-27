import os
import argparse
import SimpleITK as sitk


RESAMPLE_METHODS = {
    "axial": ["", "rotx180", "roty180", "rotz90", "rotz180", "rotz270"],
    "coronal": ["rotx90", "rotx270", "rotx270y180", "rotx90z90", "rotx90z180", "rotx90z270"],
    "sagittal": ["roty90", "roty270", "rotx180y90", "rotx180y90z90", "roty90z180", "rotx180y270z270"]
}

def add_args(parser):
    parser.add_argument("-imdir", type=str, help="Input image directory")
    parser.add_argument("-imname", type=str, help="Input image name")
    parser.add_argument("-outdir", type=str, help="Output for screenshot", default=None)

    return parser

def main(args):
    import napari
    
    results = {}
    for k, vals in RESAMPLE_METHODS.items():
        results[k] = []
        for v in vals:
            img = sitk.ReadImage(os.path.join(args.imdir, f"images3d{v}_iso3", args.imname))
            im = sitk.GetArrayFromImage(img)
            results[k].append(im)
            
    viewer = napari.view_image(results['axial'][0], name="axial0", colormap='magma')
    
    translate_value = 80
    for i, v in enumerate(results['axial'][1:]):
        viewer.add_image(v, 
                         name=f"axial{i+1}", 
                        #  translate=(0, translate_value*(i+1), 0),
                         translate=(0, 0, translate_value*(i+1)),
                         colormap='magma'
                         )
        
    for i, v in enumerate(results['coronal']):
        viewer.add_image(v, 
                         name=f"coronal{i}",
                        #  translate=[translate_value, translate_value*i, 0],
                         translate=[0, translate_value, translate_value*i],
                         colormap='magma'
                         )
    
    for i, v in enumerate(results['sagittal']):
        viewer.add_image(v, 
                         name=f"sagittal{i}",
                        #  translate=[translate_value*2, translate_value*i, 0],
                         translate=[0, translate_value*2, translate_value*i],
                         colormap='magma'
                         )
        
    # set theme
    viewer.theme = 'system'
    viewer.dims.ndisplay = 3
    
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
        viewer.screenshot(os.path.join(args.outdir, 'views3d.png'))
    
    napari.run()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Visualize augmented data')
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)