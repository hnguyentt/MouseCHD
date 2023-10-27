"""
Figures: 1b, 1c
"""

import logging
import os
import datetime
import pandas as pd
from scipy.stats import chi2_contingency

from mousechd.utils.tools import set_logger
from mousechd.utils.analyzer import (load_metadata,
                                     get_kingdom_df,
                                     plot_kingdom_venn,
                                     plot_contingency)


def add_args(parser):
    parser.add_argument('-term_path', type=str, help='Path to terminology dataframe')
    parser.add_argument('-meta_path', type=str, help='Path to metadata file')
    parser.add_argument('-sep', type=str, help='Delimeter for csv file', default=";")
    parser.add_argument('-venn_kwargs', type=dict, help='kwargs for plotting venn diagram', default={})
    parser.add_argument('-cont_kwargs', type=dict, help='kwargs for plotting contingency matrix', default={})
    parser.add_argument('-savedir', type=str, help='Save directory')
    
def main(args):
    today = datetime.datetime.now().strftime('%Y%m%d')
    os.makedirs(args.savedir, exist_ok=True)
    set_logger(os.path.join(args.savedir, f"{today}_EDA.txt"))
    df = load_metadata(args.meta_path)
    terms = pd.read_csv(args.term_path, sep=args.sep)
    
    ############################
    # Venn diagram (Figure 1b) #
    ############################
    logging.info('Plot kingdom venn diagram')
    save = os.path.join(args.savedir, f"{today}_kingdom_venn.svg")
    
    kingdom_df = get_kingdom_df(terms, df)
    plot_kingdom_venn(kingdom_df, save, **args.venn_kwargs)
    logging.info(f'Venn diagram saved in {save}')
    
    ###########################
    # Contingency (Figure 1c) #
    ###########################
    df['Diagnosis'] = df['Normal heart'].map({1: 'Normal', 0: 'CHD'})
    save = os.path.join(args.savedir, f"{today}_diagnosis_stage_contingency.svg")
    plot_contingency(df, x="Diagnosis", y="Stage",
                     save=save,
                     **args.cont_kwargs)
    logging.info(f'Diagnosis stage contingency matrix saved in {save}')
    
    ######################################
    # Chi2 test (analysis for Figure 1c) #
    ######################################
    logging.info('\nChi2 test: ')
    tb = pd.crosstab(df.Stage, df.Diagnosis)
    ChiSqResult = chi2_contingency(tb)
    logging.info("N = {}".format(tb.values.sum()))
    logging.info(f'Result: {ChiSqResult}')
    
    
    