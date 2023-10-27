def main():
    import argparse, os
    parser = argparse.ArgumentParser(description=__doc__)
    import mousechd
    parser.add_argument('-version', action='version', version=mousechd.__version__)
    
    import mousechd.run.postprocess_nnUNet
    import mousechd.run.prepare_nnUNet_data
    import mousechd.run.preprocess
    import mousechd.run.segment
    import mousechd.run.resample
    import mousechd.run.split_data
    import mousechd.run.viz3d_views
    import mousechd.run.viz3d_stages
    import mousechd.run.viz_stacks
    import mousechd.run.viz_eda
    import mousechd.run.viz3d_seg
    import mousechd.run.create_label_df
    import mousechd.run.test_clf
    import mousechd.run.train_clf
    import mousechd.run.explain
    import mousechd.run.viz_grad
    
    modules = [
        mousechd.run.postprocess_nnUNet,
        mousechd.run.prepare_nnUNet_data,
        mousechd.run.preprocess,
        mousechd.run.segment,
        mousechd.run.resample,
        mousechd.run.split_data,
        mousechd.run.viz3d_views,
        mousechd.run.viz3d_stages,
        mousechd.run.viz_stacks,
        mousechd.run.viz_eda,
        mousechd.run.viz3d_seg,
        mousechd.run.create_label_df,
        mousechd.run.test_clf,
        mousechd.run.train_clf,
        mousechd.run.explain,
        mousechd.run.viz_grad
    ]
    
    subparsers = parser.add_subparsers(title='Choose a command', required=True)
    
    def get_str_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]
    
    for module in modules:
        this_parser = subparsers.add_parser(get_str_name(module), description=module.__doc__)
        module.add_args(parser=this_parser)
        this_parser.set_defaults(func=module.main)
        
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()