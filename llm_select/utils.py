import os.path as osp
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from llm_select import datasets

ABS_DIR = osp.dirname(osp.abspath(__file__))

DATAPREDS = [
    # Classification
    'credit-g/risk',
    'bank/subscription',
    'give-me-credit/delinquency',
    'compas/recid',
    'pima/diabetes',
    
    # Regression
    'calhousing/price',
    'diabetes/progression',
    'wine/quality',
    'miami-housing/price',
    'cars/price'
]

LARGE_DATAPREDS = [
    'acs/income',
    'acs/employment',
    'acs/public_coverage',
    'acs/mobility',
    'mimic-icd/hf',
    'mimic-icd/ckd',
    'mimic-icd/copd'
]

DATASET_TO_CLASS = {
    'cars': datasets.CarsDataset,
    'pima': datasets.PimaDataset,
    'give-me-credit': datasets.GiveMeCreditDataset,
    'compas': datasets.COMPASDataset,
    'miami-housing': datasets.MiamiHousingDataset,
    'wine': datasets.WineDataset,
    'credit-g': datasets.CreditGDataset,
    'diabetes': datasets.DiabetesDataset,
    'calhousing': datasets.CalHousingDataset,
    'bank': datasets.BankDataset,
    'acs': datasets.ACSDataset,
    'mimic-icd': datasets.MIMICICDDataset,
}

LLM_TO_PLOTNAME = {
    'gpt-4-0613': 'GPT-4 (1.7T)',
    'gpt-3.5-turbo': 'GPT-3.5 (175B)',
    'llama-2-70b-chat': 'Llama-2 (70B)',
    'llama-2-13b-chat': 'Llama-2 (13B)',
    'llama-2-7b-chat': 'Llama-2 (7B)'
}

LLM_TO_INDEX = {
    'gpt-4-0613': 4,
    'GPT-4 (1.7T)': 4,
    'gpt-3.5-turbo': 3,
    'GPT-3.5 (175B)': 3,
    'llama-2-70b-chat': 2,
    'Llama-2 (70B)': 2,
    'llama-2-13b-chat': 1,
    'Llama-2 (13B)': 1,
    'llama-2-7b-chat': 0,
    'Llama-2 (7B)': 0
}

PROMPTMETHOD_TO_PLOTNAME = {    
    # LLM suffixes
    'greedy::': 'T=0 (Default)*',
    'greedy::examples': 'T=0 (Examples)',
    'greedy::expls': 'T=0 (Examples with CoT)',
    'greedy:context:': 'T=0 (Context)',
    'greedy:context:examples': 'T=0 (Context + Examples)',
    'greedy:context:expls': 'T=0 (Context + Examples with CoT)',
    
    'consistent::': 'T=0.5',
    'consistent::examples': 'T=0.5 (Examples)',
    'consistent::expls': 'T=0.5 (Examples with CoT)',
    'consistent:context:': 'T=0.5 (Context)',
    'consistent:context:examples': 'T=0.5 (Context + Examples)',
    'consistent:context:expls': 'T=0.5 (Context + Examples with CoT)'
}

FSMETHOD_TO_PLOTNAME = {
    'lassonet': 'LassoNet',
    'lasso': 'LASSO',
    'forward-seq': 'Forward',
    'backward-seq': 'Backward',
    'mrmr': 'MRMR',
    'filter-mi': 'Mutual Information',
    'rfe': 'RFE',
    'random-concept': 'Random'
}

FIMETHOD_TO_PLOTNAME = {
    'range_1': 'Score Range: [0,10]',
    'range_2': 'Score Range: [8,24]',
    'rank': 'LLM-Rank',
    'forward-seq': 'LLM-Seq',
    'forward-seq-empty': 'LLM-Seq',
    'shap-linear': 'SHAP: Linear',
    'shap-xgb': 'SHAP: XGBoost',
    'mi': 'Mutual Information',
    'pearson': 'Pearson',
    'spearman': 'Spearman',
    'fisher': 'Fisher Score',
    'perm': 'Permutation',
    'random': 'Random'
}

DATAPRED_TO_PLOTNAME = {
    'mimic-icd/hf': 'HF',
    'mimic-icd/ckd': 'CKD',
    'mimic-icd/copd': 'COPD',
    'acs/employment': 'Employment',
    'acs/income': 'Income',
    'acs/mobility': 'Mobility',
    'acs/public_coverage': 'Public Coverage',
    'bank/subscription': 'Bank',
    'calhousing/price': 'CA Housing',
    'diabetes/progression': 'Diabetes Progression',
    'credit-g/risk': 'Credit-G',
    'wine/quality': 'Wine Quality',
    'miami-housing/price': 'Miami Housing',
    'compas/recid': 'COMPAS Recidivism',
    'give-me-credit/delinquency': 'Give Me Some Credit',
    'pima/diabetes': 'Pima Indians Diabetes',
    'cars/price': 'Used Cars'
}

N_TO_SELECT = {
    # MIMIC HF
    'mimic-icd/hf/0.3': 812,
    
    # MIMIC CKD
    'mimic-icd/ckd/0.3': 788,
    
    # MIMIC COPD
    'mimic-icd/copd/0.3': 860,
    
    # ACS Income
    'acs/income/0.3': 2026,

    # ACS Employment
    'acs/employment/0.3': 499,

    # ACS Public Coverage
    'acs/public_coverage/0.3': 2042,

    # ACS Mobility
    'acs/mobility/0.3': 1853,
}

def plot_perf_vs_ratio(
    datapreds=DATAPREDS,
    figsize=(40,13),
    llm_select_mode='llm-ratio',
    exp_suffix='baseline',
    outdir=osp.join(ABS_DIR, 'results/linear_compare'),
    add_legend=True,
    add_std=False,
    save_png=False,
    save_pdf=False,
):
    '''Generates a plot of the feature selection path.'''

    sns.set_style('whitegrid')
    palette = [ 
        '#BCBD22', '#9A6324', '#469990', '#3E454B', '#797FEF', '#F032E6', '#8D1C3D', '#000000', # Baselines
        '#911EB4', '#F58231', '#4363D8', '#3CB44B', '#E8000B' # LLM feature selection methods
    ]
    sns.set_palette(palette)
    fig, axes = plt.subplots(2, 5, figsize=figsize)

    for i, datapred in enumerate(datapreds):
        dataset, pred = datapred.split('/')
        res_path = osp.join(outdir, f'{dataset}/{pred}_results')
        res_path += f'_{exp_suffix}' if exp_suffix != '' else ''
        res_path += '.pkl'
        
        with open(res_path, 'rb') as fh:
            res = pickle.load(fh)
            
        row, col = i//5, i%5
        baseline_config = dict(
            markersize=14,
            linestyle=(0,(1,1)), # Densely dotted
            marker='o',
            mfc='none',
            linewidth=3
        )
        random_config = dict(
            markersize=14,
            linestyle=(0,(5,1)), # 'dashed'
            marker='^',
            mfc='none',
            linewidth=3
        )
        llm_config = dict(
            markersize=14,
            linestyle='solid',
            marker='v',
            mfc='none',
            linewidth=4
        )

        # LassoNet
        v = res['lassonet']
        axes[row][col].plot(v[0], v[1], label='LassoNet', **baseline_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # LASSO
        v = res['lasso']
        axes[row][col].plot(v[0], v[1], label='LASSO', **baseline_config)
        
        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # Forward Selection
        v = res['forward-seq']
        axes[row][col].plot(v[0], v[1], label='Forward', **baseline_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # Backward Selection
        v = res['backward-seq']
        axes[row][col].plot(v[0], v[1], label='Backward', **baseline_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # RFE
        v = res['rfe']
        axes[row][col].plot(v[0], v[1], label='RFE', **baseline_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # MRMR
        v = res['mrmr']
        axes[row][col].plot(v[0],v[1], label='MRMR', **baseline_config)

        if add_std:
            axes[row][col].fill_between(v[0],v[1]-v[2], v[1]+v[2], alpha=0.1)

        # Mutual Information
        v = res['filter-mi']
        axes[row][col].plot(v[0], v[1], label='Mutual Information', **baseline_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # Random
        v = res['random-concept']
        axes[row][col].plot(v[0], v[1], label='Random', **random_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # Llama-2-7B Greedy
        v = res[f'{llm_select_mode}:llama-2-7b-chat:greedy::']
        axes[row][col].plot(v[0], v[1], label='Llama-2 (7B)', **llm_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)
    
        # Llama-2-13B Greedy
        v = res[f'{llm_select_mode}:llama-2-13b-chat:greedy::']
        axes[row][col].plot(v[0], v[1], label='Llama-2 (13B)', **llm_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # Llama-2-70B Greedy
        v = res[f'{llm_select_mode}:llama-2-70b-chat:greedy::']
        axes[row][col].plot(v[0], v[1], label='Llama-2 (70B)', **llm_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)
        
        # GPT-3.5 Greedy
        v = res[f'{llm_select_mode}:gpt-3.5-turbo:greedy::']
        axes[row][col].plot(v[0], v[1], label='GPT-3.5 (175B)', **llm_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        # GPT-4 Greedy
        v = res[f'{llm_select_mode}:gpt-4-0613:greedy::']
        axes[row][col].plot(v[0], v[1], label='GPT-4 (1.7T)', **llm_config)

        if add_std:
            axes[row][col].fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.1)

        axes[row][col].set_xlabel('Fraction Selected', fontsize=30, labelpad=10)
        axes[row][col].set_ylabel(('AUROC' if row == 0 else 'MAE'), fontsize=30, labelpad=10)
        axes[row][col].set_title(DATAPRED_TO_PLOTNAME[datapred], fontsize=30)
        axes[row][col].set_xticks([0.2,0.4,0.6,0.8,1.0])
        axes[row][col].tick_params(axis='both', labelsize=25)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    if add_legend:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='center',
            bbox_to_anchor=(0.5, 1),
            ncol=7,
            bbox_transform=fig.transFigure,
            fontsize=30
        )

    if save_png:
        plot_path = osp.join(outdir, f'perf_vs_ratio_plot_{llm_select_mode}.png')
        plt.savefig(plot_path, bbox_inches='tight')
    elif save_pdf:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman"
        })

        plot_path = osp.join(outdir, f'perf_vs_ratio_plot_{llm_select_mode}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', backend='pgf', dpi=1600)
    else:
        plt.show()

def plot_perf_vs_ratio_llm(
    datapreds=DATAPREDS,
    llm_models=['gpt-4-0613'],
    figsize=(40,13),
    exp_suffix='baseline',
    outdir=osp.join(ABS_DIR, 'results/linear_compare'),
    save_png=False,
    save_pdf=False,
    add_std=False,
    show_seq_results=False,
    out_suffix=''
):
    '''Generates a performance vs. fraction of features selected plot for each LLM.'''

    sns.set_style('whitegrid')
    palette = sns.color_palette('tab10')
    fig, axes = plt.subplots(2, 5, figsize=figsize)

    for i, datapred in enumerate(datapreds):
        dataset, pred = datapred.split('/')
        res_path = osp.join(outdir, f'{dataset}/{pred}_results')
        res_path += f'_{exp_suffix}' if exp_suffix != '' else ''
        res_path += '.pkl'
        
        with open(res_path, 'rb') as fh:
            res = pickle.load(fh)
            
        row, col = i//5, i%5
        base_config = dict(
            markersize=20,
            linestyle='solid',
            mfc='none',
            linewidth=4
        )
        greedy_config = dict(marker='v')
        seq_config = dict(marker='s')
        seq_empty_config = dict(marker='p')
        rank_config = dict(marker='d')

        for llm_model in llm_models:
            if show_seq_results:
                # Sequential (Empty)
                v = res[f'llm-forward-seq-empty:{llm_model}:greedy::']
                axes[row][col].plot(
                    v[0], v[1],
                    label=f'{LLM_TO_PLOTNAME[llm_model]}, Sequential (Empty)',
                    color=palette[1],
                    **base_config,
                    **seq_empty_config
                )
                if add_std:
                    axes[row][col].fill_between(
                        v[0], v[1]-v[2], v[1]+v[2], alpha=0.1, color=palette[1]
                    )

                # Sequential (Nonempty)
                v = res[f'llm-forward-seq:{llm_model}:greedy::']
                axes[row][col].plot(
                    v[0], v[1],
                    label=f'{LLM_TO_PLOTNAME[llm_model]}, Sequential (LLM-Score)',
                    color=palette[4],
                    **base_config,
                    **seq_config
                )
                if add_std:
                    axes[row][col].fill_between(
                        v[0], v[1]-v[2], v[1]+v[2], alpha=0.1, color=palette[4]
                    )
            else:
                # Score
                v = res[f'llm-ratio:{llm_model}:greedy::']
                axes[row][col].plot(
                    v[0], v[1], 
                    label=f'{LLM_TO_PLOTNAME[llm_model]}, Score', 
                    color=palette[0],
                    **base_config, 
                    **greedy_config
                )
                if add_std:
                    axes[row][col].fill_between(
                        v[0], v[1]-v[2], v[1]+v[2], alpha=0.1, color=palette[0]
                    )

                # Sequential
                v = res[f'llm-forward-seq-empty:{llm_model}:greedy::']
                axes[row][col].plot(
                    v[0], v[1],
                    label=f'{LLM_TO_PLOTNAME[llm_model]}, Sequential',
                    color=palette[1],
                    **base_config,
                    **seq_config
                )
                if add_std:
                    axes[row][col].fill_between(
                        v[0], v[1]-v[2], v[1]+v[2], alpha=0.1, color=palette[1]
                    )

                # Rank
                v = res[f'llm-rank:{llm_model}:greedy::']
                axes[row][col].plot(
                    v[0], v[1],
                    label=f'{LLM_TO_PLOTNAME[llm_model]}, Rank',
                    color=palette[2],
                    **base_config,
                    **rank_config
                )
                if add_std:
                    axes[row][col].fill_between(
                        v[0], v[1]-v[2], v[1]+v[2], alpha=0.1, color=palette[2]
                    )

        axes[row][col].set_xlabel('Fraction Selected', fontsize=30, labelpad=10)
        axes[row][col].set_ylabel(('AUROC' if row == 0 else 'MAE'), fontsize=30, labelpad=10)
        axes[row][col].set_title(DATAPRED_TO_PLOTNAME[datapred], fontsize=30)
        axes[row][col].set_xticks([0.2,0.4,0.6,0.8,1.0])
        axes[row][col].tick_params(axis='both', labelsize=25)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center',
        bbox_to_anchor=(0.5, 1),
        ncol=5,
        bbox_transform=fig.transFigure,
        fontsize=30
    )

    if save_png:
        plot_path = osp.join(outdir, 'perf_vs_ratio_llm_plot')
        plot_path += f'_seq' if show_seq_results else ''
        plot_path += f'_{out_suffix}' if out_suffix != '' else ''
        plot_path += '.png'
        plt.savefig(plot_path, bbox_inches='tight')

    elif save_pdf:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman"
        })

        plot_path = osp.join(outdir, 'perf_vs_ratio_llm_plot')
        plot_path += f'_seq' if show_seq_results else ''
        plot_path += f'_{out_suffix}' if out_suffix != '' else ''
        plot_path += '.pdf'
        plt.savefig(plot_path, bbox_inches='tight', backend='pgf', dpi=1600)
    
    else:
        plt.show()

def summarize_perf_vs_ratio_auc(
    datapreds=DATAPREDS,
    llm_select_mode='llm-ratio',
    exp_suffix='baseline',
    results_dir=osp.join(ABS_DIR, 'results/linear_compare'),
    save_latex=False,
    return_df=False
):
    '''Computes the AUC for the feature selection paths.'''

    auc_dict = {'Dataset': [], 'Selection': [], 'Area': []}

    baselines = [
        'lassonet', 'lasso', 'forward-seq', 'backward-seq', 'rfe', 'mrmr', 'filter-mi', 'random-concept'
    ]
    llms = [
        f'{llm_select_mode}:{llm_model}:greedy::' for llm_model in [
            'gpt-4-0613', 
            'gpt-3.5-turbo', 
            'llama-2-70b-chat', 
            'llama-2-13b-chat', 
            'llama-2-7b-chat'
        ]
    ]
    fs_methods = baselines + llms

    for i, datapred in enumerate(datapreds):
        dataset, pred = datapred.split('/')
        res_path = osp.join(results_dir, f'{dataset}/{pred}_results')
        res_path += f'_{exp_suffix}' if exp_suffix != '' else ''
        res_path += '.pkl'

        with open(res_path, 'rb') as fh:
            res = pickle.load(fh)

        # Get prediction type
        pred_type = DATASET_TO_CLASS[dataset].get_pred_type()

        for fs_method in fs_methods:
            v = res[fs_method]
            sorted_idxs = np.argsort(v[0])
            fracs = v[0][sorted_idxs]
            test_metrics = v[1][sorted_idxs]
            
            if pred_type == 'classification':
                fracs = np.insert(fracs, 0, 0)
                test_metrics = np.insert(test_metrics, 0, 0.5)
            
            elif pred_type == 'regression':
                fracs = np.insert(fracs, 0, 0)
                test_metrics = np.insert(test_metrics, 0, test_metrics[0])

            auc = np.trapz(test_metrics, fracs)
            auc_dict['Dataset'].append(DATAPRED_TO_PLOTNAME[datapred])
            auc_dict['Area'].append(auc)

            if fs_method in llms:
                auc_dict['Selection'].append(
                    LLM_TO_PLOTNAME[fs_method.split(':')[1]]
                )
            else:
                auc_dict['Selection'].append(FSMETHOD_TO_PLOTNAME[fs_method])

    df = pd.DataFrame(auc_dict)
    df['Area'] = df['Area'].apply(lambda x: f'{x:.4f}')
    df = df.pivot_table(index='Selection', columns='Dataset', values='Area')
    df = df[[
        'Credit-G', 'Bank', 'Give Me Some Credit', 'COMPAS Recidivism', 'Pima Indians Diabetes',
        'CA Housing', 'Diabetes Progression', 'Wine Quality', 'Miami Housing', 'Used Cars'
    ]]
    col_tuples = [
        ('Classification', col) for col in df.columns
        if col in ['Credit-G', 'Bank', 'Give Me Some Credit', 'COMPAS Recidivism', 'Pima Indians Diabetes']
    ]
    col_tuples += [
        ('Regression', col) for col in df.columns
        if col in ['CA Housing', 'Diabetes Progression', 'Wine Quality', 'Miami Housing', 'Used Cars']
    ]
    df.columns = pd.MultiIndex.from_tuples(col_tuples)
    df.index.name = ''
    df = df.loc[[
        'LassoNet',
        'LASSO',
        'Forward',
        'Backward',
        'MRMR',
        'Mutual Information',
        'RFE',
        'Random',
        'GPT-4 (1.7T)',
        'GPT-3.5 (175B)',
        'Llama-2 (70B)',
        'Llama-2 (13B)',
        'Llama-2 (7B)'
    ]]
    
    table = df.to_latex(
        position='t!',
        column_format='@{}' + 'c'*11 + '@{}',
        multicolumn_format='c',
        caption=r'Area under the feature selection path for all small-scale datasets.',
        label='tab:small-exp-auc'
    )

    if save_latex:
        tex_path = osp.join(results_dir, 'small_exp_auc.tex')
        with open(tex_path, 'w') as fh:
            fh.write(table)
        
        print(f'Small-dataset experiment AUC table saved as .tex file in: {tex_path}')
    
    if return_df:
        return df
    else:
        return table

def plot_auc_pct_change(
    datapreds=DATAPREDS, 
    figsize=(30,10),
    ylim=(-45,45),
    plot_type='box',
    errorbar='sd',
    exp_suffix='baseline', 
    outdir=osp.join(ABS_DIR, 'results/linear_compare'),
    save_png=False,
    save_pdf=False
):
    '''Generates a box/bar plot of percent change in area under feature selection paths across prompting strategies.'''

    llm_aucs = {'datapred': [], 'llm': [], 'prompt_method': [], 'auc_pct_change': []}
    for datapred in datapreds:
        dataset, pred = datapred.split('/')
        
        # Get prediction type
        pred_type = DATASET_TO_CLASS[dataset].get_pred_type()
        
        # Load the results
        res_path = osp.join(outdir, f'{dataset}/{pred}_results')
        res_path += f'_{exp_suffix}' if exp_suffix != '' else ''
        res_path += '.pkl'

        with open(res_path, 'rb') as fh:
            res = pickle.load(fh)

        # Retrieve the evaluated LLM model names
        llms = sorted(list(set([k.split(':')[1] for k in res.keys() if 'llm' in k])))
        all_llms = [
            'gpt-4-0613', 'gpt-3.5-turbo', 'llama-2-70b-chat', 'llama-2-13b-chat', 'llama-2-7b-chat'
        ]

        # Compute the AUC or (1-AUC) percent changes
        for llm in all_llms:
            if llm not in llms:
                continue

            llm_res = {k:v for k,v in res.items() if llm in k}

            # Retrieve greedy decoding baseline
            llm_base = llm_res[f'llm-ratio:{llm}:greedy::']

            if pred_type == 'classification':
                llm_base_fracs = np.insert(llm_base[0], 0, 0)
                llm_base_test_metrics = np.insert(llm_base[1], 0, 0.5)
            
            elif pred_type == 'regression':
                llm_base_fracs = np.insert(llm_base[0], 0, 0)
                llm_base_test_metrics = np.insert(llm_base[1], 0, llm_base[1][0])

            # Area under feature selection path
            llm_base_auc = np.trapz(llm_base_test_metrics, llm_base_fracs) 

            # Calculate percent change in all other prompting strategies
            for k,v in llm_res.items():
                if 'llm-rank' in k or 'llm-forward-seq' in k:
                    continue

                sorted_idxs = np.argsort(v[0])
                fracs = v[0][sorted_idxs]
                test_metrics = v[1][sorted_idxs]

                # Calculate percent increase
                if pred_type == 'classification':
                    fracs = np.insert(fracs, 0, 0)
                    test_metrics = np.insert(test_metrics, 0, 0.5)
                    auc = np.trapz(test_metrics, fracs)
                    auc_pct_change = ((llm_base_auc - auc) / llm_base_auc) * 100
                
                # Calculate percent decrease
                elif pred_type == 'regression':
                    fracs = np.insert(fracs, 0, 0)
                    test_metrics = np.insert(test_metrics, 0, test_metrics[0])
                    auc = np.trapz(test_metrics, fracs)
                    auc_pct_change = ((auc - llm_base_auc) / llm_base_auc) * 100
                
                fs_split = k.split(':')
                model, prompt_method = fs_split[1], ':'.join(fs_split[2:])
                llm_aucs['datapred'].append(datapred)
                llm_aucs['llm'].append(model)
                llm_aucs['prompt_method'].append(prompt_method)
                llm_aucs['auc_pct_change'].append(auc_pct_change)

    auc_df = pd.DataFrame(llm_aucs)
    auc_df['llm'] = auc_df['llm'].apply(lambda x: LLM_TO_PLOTNAME[x])
    auc_df['prompt_method'] = auc_df['prompt_method'].apply(lambda x: PROMPTMETHOD_TO_PLOTNAME[x])
    auc_df.rename(
        columns={'llm': 'LLM', 'prompt_method': 'Decoding (Prompt Design)', 'auc_pct_change': '%AUC'}, 
        inplace=True
    )

    # Separate color palette for greedy and self-consistency decoding
    palette = list(sns.color_palette('rocket', 6).as_hex()) + list(sns.color_palette('viridis', 6).as_hex())
    palette = {k: v for (k,v) in zip(PROMPTMETHOD_TO_PLOTNAME.values(), palette)}

    sns.set_style('whitegrid')
    _ = plt.figure(figsize=figsize)

    # NOTE: For bar plot, set ylim=(-20,20)
    if plot_type == 'bar':
        ax = sns.barplot(
            data=auc_df, 
            x='LLM', 
            y='%AUC', 
            hue='Decoding (Prompt Design)', 
            errorbar=errorbar, 
            errwidth=0.5, 
            palette=palette
        )
        ax.set_ylabel(r'Average Improvement (%)', fontsize=40)
    
    # NOTE: For box plot, set ylim=(-45,45)
    elif plot_type == 'box':
        ax = sns.boxplot(
            data=auc_df,
            x='LLM',
            y='%AUC',
            hue='Decoding (Prompt Design)',
            palette=palette,
            fliersize=12.5,
            flierprops={'marker': 'x'},
            linewidth=3.5,
        )
        ax.set_ylabel(r'Improvement in Selection (%)', fontsize=35)
        
    ax.tick_params(axis='both', labelsize=40)
    ax.axhline(y=0, linestyle='dashed', linewidth=4, color='black')
    ax.set_ylim(*ylim)
    sns.move_legend(
        ax, 
        loc='lower left',
        bbox_to_anchor=(1.03,-0.05),
        ncol=1, 
        fontsize=30, 
        framealpha=1,
        title=''
    )
    ax.set_xlabel('')

    # Bold face the baseline prompting setup
    legend = ax.legend_
    handles, labels = ax.get_legend_handles_labels()
    for _, label in zip(handles, labels):
        if label == 'T=0 (Default)*':
            legend.get_texts()[labels.index(label)].set_weight('bold')

    if save_png:
        plot_path = osp.join(outdir, f'prompt_auc_pct_change_{plot_type}.png')
        plt.savefig(plot_path, bbox_inches='tight')
    
    elif save_pdf:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman"
        })

        plot_path = osp.join(outdir, f'prompt_auc_pct_change_{plot_type}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', backend='pgf', dpi=1600)
    
    else:
        plt.show()

def plot_rank_corr(
    datapreds=DATAPREDS,
    corr_metrics=[
        'shap-xgb',
        'fisher',
        'mi',
        'pearson',
        'spearman',
        'perm'
    ],
    llm_on_xaxis=False,
    figsize=(25,5),
    ylim=(-1,1),
    plot_type='bar',
    errorbar='sd',
    results_dir=osp.join(ABS_DIR, 'results/rank_corr'),
    save_png=False,
    save_pdf=False
):
    '''Generates a rank correlation line/box plot.'''

    # Load the rank correlations dataframe
    rank_corr_df = pd.read_csv(osp.join(results_dir, 'rank_corr.csv'))
    rank_corr_df = rank_corr_df[rank_corr_df['datapred'].isin(datapreds)]
    rank_corr_df['llm_model'] = rank_corr_df['llm_model'].apply(lambda x: LLM_TO_PLOTNAME[x])
    rank_corr_df['fi_method'] = rank_corr_df['fi_method'].apply(lambda x: FIMETHOD_TO_PLOTNAME[x])
    rank_corr_df.rename(columns={'llm_model': 'LLM', 'fi_method': 'Importance'}, inplace=True)

    imp_order = [FIMETHOD_TO_PLOTNAME[c] for c in corr_metrics]
    imp_order = [m for m in imp_order if m in rank_corr_df['Importance'].unique()]

    if llm_on_xaxis:
        x, hue = 'LLM', 'Importance'
        order = ['GPT-4 (1.7T)', 'GPT-3.5 (175B)', 'Llama-2 (70B)', 'Llama-2 (13B)', 'Llama-2 (7B)']
        hue_order = imp_order
        ncol = 4
    else:
        x, hue = 'Importance', 'LLM'
        order = imp_order
        hue_order = ['GPT-4 (1.7T)', 'GPT-3.5 (175B)', 'Llama-2 (70B)', 'Llama-2 (13B)', 'Llama-2 (7B)']
        ncol = 5

    sns.set_style('whitegrid')
    sns.set_palette('tab10')
    plt.figure(figsize=figsize)
    plt.axhline(y=0, linestyle='dashed', linewidth=2, color='black')

    if plot_type == 'bar':
        ax = sns.barplot(
            data=rank_corr_df, 
            x=x, 
            y='kendall_tau', 
            hue=hue,
            order=order,
            hue_order=hue_order,
            errorbar=errorbar,
            errwidth=0.5,
            width=1.5
        )
        ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5,0), ncol=ncol, fontsize=23, title='')
        ax.set_ylabel('Average Rank Correlation', fontsize=23)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=23)

    elif plot_type == 'box':
        ax = sns.boxplot(
            data=rank_corr_df, 
            x=x, 
            y='kendall_tau', 
            hue=hue,
            order=order,
            hue_order=hue_order,
            fliersize=10,
            flierprops={'marker': 'x'}
        )
        ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower left', bbox_to_anchor=(0,0), ncol=4, fontsize=20, title='')
        ax.set_ylabel('Rank Correlation', fontsize=20)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=20)

    elif plot_type == 'line':
        rank_corr_df['LLM Index'] = rank_corr_df['LLM'].apply(lambda x: LLM_TO_INDEX[x])
        ax = sns.lineplot(
            data=rank_corr_df,
            x='LLM Index',
            y='kendall_tau',
            hue='Importance',
            hue_order=hue_order,
            errorbar=None,
            marker='v',
            markersize=30,
            mfc='none'
        )
        #ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower left', bbox_to_anchor=(0,0), ncol=4, fontsize=20, title='')
        ax.set_ylabel('Rank Correlation', fontsize=20)
        ax.set_xlabel('Model Scale')
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(['7B', '13B', '70B', '175B', '1.7T'])
        ax.tick_params(axis='both', labelsize=20)

    if save_png:
        plot_path = osp.join(results_dir, f'rank_corr_{plot_type}.png')
        plt.savefig(plot_path, bbox_inches='tight')

    elif save_pdf:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman"
        })

        plot_path = osp.join(results_dir, f'rank_corr_{plot_type}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', backend='pgf', dpi=1600)

    else:
        plt.show()

def plot_rank_corr_score_range(
    datapreds=DATAPREDS,
    corr_metrics=[
        'range_1',
        'range_2'
    ],
    llm_on_xaxis=False,
    figsize=(12,5),
    ylim=(-1,1),
    plot_type='bar',
    errorbar='sd',
    results_dir=osp.join(ABS_DIR, 'results/rank_corr'),
    save_png=False,
    save_pdf=False
):
    '''Generates a rank correlation line/box plot for the different score ranges.'''

    # Load the rank correlations dataframe
    rank_corr_df = pd.read_csv(osp.join(results_dir, 'rank_corr.csv'))
    rank_corr_df = rank_corr_df[rank_corr_df['datapred'].isin(datapreds)]
    rank_corr_df['llm_model'] = rank_corr_df['llm_model'].apply(lambda x: LLM_TO_PLOTNAME[x])
    rank_corr_df['fi_method'] = rank_corr_df['fi_method'].apply(lambda x: FIMETHOD_TO_PLOTNAME[x])
    rank_corr_df.rename(columns={'llm_model': 'LLM', 'fi_method': 'Importance'}, inplace=True)

    imp_order = [FIMETHOD_TO_PLOTNAME[c] for c in corr_metrics]
    imp_order = [m for m in imp_order if m in rank_corr_df['Importance'].unique()]

    if llm_on_xaxis:
        x, hue = 'LLM', 'Importance'
        order = ['GPT-4 (1.7T)', 'GPT-3.5 (175B)', 'Llama-2 (70B)', 'Llama-2 (13B)', 'Llama-2 (7B)']
        hue_order = imp_order
        ncol = 4
    else:
        x, hue = 'Importance', 'LLM'
        order = imp_order
        hue_order = ['GPT-4 (1.7T)', 'GPT-3.5 (175B)', 'Llama-2 (70B)', 'Llama-2 (13B)', 'Llama-2 (7B)']
        ncol = 3

    sns.set_style('whitegrid')
    sns.set_palette('tab10')
    plt.figure(figsize=figsize)
    plt.axhline(y=0, linestyle='dashed', linewidth=2, color='black')

    if plot_type == 'bar':
        ax = sns.barplot(
            data=rank_corr_df, 
            x=x, 
            y='kendall_tau', 
            hue=hue,
            order=order,
            hue_order=hue_order,
            errorbar=errorbar,
            errwidth=0.5,
            width=0.8
        )
        ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5,0), ncol=ncol, fontsize=19, title='')
        ax.set_ylabel('Average Rank Correlation', fontsize=23)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=23)

    elif plot_type == 'box':
        ax = sns.boxplot(
            data=rank_corr_df, 
            x=x, 
            y='kendall_tau', 
            hue=hue,
            order=order,
            hue_order=hue_order,
            fliersize=10,
            flierprops={'marker': 'x'}
        )
        ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower left', bbox_to_anchor=(0,0), ncol=4, fontsize=20, title='')
        ax.set_ylabel('Rank Correlation', fontsize=20)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=20)

    elif plot_type == 'line':
        rank_corr_df['LLM Index'] = rank_corr_df['LLM'].apply(lambda x: LLM_TO_INDEX[x])
        ax = sns.lineplot(
            data=rank_corr_df,
            x='LLM Index',
            y='kendall_tau',
            hue='Importance',
            hue_order=hue_order,
            errorbar=None,
            marker='v',
            markersize=30,
            mfc='none'
        )
        #ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower left', bbox_to_anchor=(0,0), ncol=4, fontsize=20, title='')
        ax.set_ylabel('Rank Correlation', fontsize=20)
        ax.set_xlabel('Model Scale')
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(['7B', '13B', '70B', '175B', '1.7T'])
        ax.tick_params(axis='both', labelsize=20)

    if save_png:
        plot_path = osp.join(results_dir, f'rank_corr_score_range_{plot_type}.png')
        plt.savefig(plot_path, bbox_inches='tight')

    elif save_pdf:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman"
        })

        plot_path = osp.join(results_dir, f'rank_corr_score_range_{plot_type}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', backend='pgf', dpi=1600)

    else:
        plt.show()

def plot_rank_corr_rank_seq(
    datapreds=DATAPREDS,
    corr_metrics=[
        'rank',
        'forward-seq'
    ],
    llm_on_xaxis=False,
    figsize=(12,5),
    ylim=(-1,1),
    plot_type='bar',
    errorbar='sd',
    results_dir=osp.join(ABS_DIR, 'results/rank_corr'),
    save_png=False,
    save_pdf=False
):
    '''Generates a rank correlation line/box plot between LLM-Score and (LLM-Rank, LLM-Seq).'''

    # Load the rank correlations dataframe
    rank_corr_df = pd.read_csv(osp.join(results_dir, 'rank_corr.csv'))
    rank_corr_df = rank_corr_df[rank_corr_df['datapred'].isin(datapreds)]
    rank_corr_df['llm_model'] = rank_corr_df['llm_model'].apply(lambda x: LLM_TO_PLOTNAME[x])
    rank_corr_df['fi_method'] = rank_corr_df['fi_method'].apply(lambda x: FIMETHOD_TO_PLOTNAME[x])
    rank_corr_df.rename(columns={'llm_model': 'LLM', 'fi_method': 'Importance'}, inplace=True)

    imp_order = [FIMETHOD_TO_PLOTNAME[c] for c in corr_metrics]
    imp_order = [m for m in imp_order if m in rank_corr_df['Importance'].unique()]

    if llm_on_xaxis:
        x, hue = 'LLM', 'Importance'
        order = ['GPT-4 (1.7T)', 'GPT-3.5 (175B)', 'Llama-2 (70B)', 'Llama-2 (13B)', 'Llama-2 (7B)']
        hue_order = imp_order
        ncol = 4
    else:
        x, hue = 'Importance', 'LLM'
        order = imp_order
        hue_order = ['GPT-4 (1.7T)', 'GPT-3.5 (175B)', 'Llama-2 (70B)', 'Llama-2 (13B)', 'Llama-2 (7B)']
        ncol = 3

    sns.set_style('whitegrid')
    sns.set_palette('tab10')
    plt.figure(figsize=figsize)
    plt.axhline(y=0, linestyle='dashed', linewidth=2, color='black')

    if plot_type == 'bar':
        ax = sns.barplot(
            data=rank_corr_df, 
            x=x, 
            y='kendall_tau', 
            hue=hue,
            order=order,
            hue_order=hue_order,
            errorbar=errorbar,
            errwidth=0.5,
            width=0.8
        )
        ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5,0), ncol=ncol, fontsize=19, title='')
        ax.set_ylabel('Average Rank Correlation', fontsize=23)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=23)

    elif plot_type == 'box':
        ax = sns.boxplot(
            data=rank_corr_df, 
            x=x, 
            y='kendall_tau', 
            hue=hue,
            order=order,
            hue_order=hue_order,
            fliersize=10,
            flierprops={'marker': 'x'}
        )
        ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower left', bbox_to_anchor=(0,0), ncol=4, fontsize=20, title='')
        ax.set_ylabel('Rank Correlation', fontsize=20)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=20)

    elif plot_type == 'line':
        rank_corr_df['LLM Index'] = rank_corr_df['LLM'].apply(lambda x: LLM_TO_INDEX[x])
        ax = sns.lineplot(
            data=rank_corr_df,
            x='LLM Index',
            y='kendall_tau',
            hue='Importance',
            hue_order=hue_order,
            errorbar=None,
            marker='v',
            markersize=30,
            mfc='none'
        )
        #ax.set_ylim(*ylim)
        sns.move_legend(ax, loc='lower left', bbox_to_anchor=(0,0), ncol=4, fontsize=20, title='')
        ax.set_ylabel('Rank Correlation', fontsize=20)
        ax.set_xlabel('Model Scale')
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(['7B', '13B', '70B', '175B', '1.7T'])
        ax.tick_params(axis='both', labelsize=20)

    if save_png:
        plot_path = osp.join(results_dir, f'rank_corr_rank_seq_{plot_type}.png')
        plt.savefig(plot_path, bbox_inches='tight')

    elif save_pdf:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman"
        })

        plot_path = osp.join(results_dir, f'rank_corr_rank_seq_{plot_type}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', backend='pgf', dpi=1600)

    else:
        plt.show()

def plot_reg_path(
    datapreds=[
        'acs/income', 
        'acs/employment', 
        'acs/public_coverage',
        'acs/mobility',
        'mimic-icd/ckd',
        'mimic-icd/copd',
        'mimic-icd/hf'
    ],
    fs_method='lassonet',
    selection_dir=osp.join(ABS_DIR, 'selection'),
    figsize=(40,6),
    save_png=False,
    save_pdf=False  
):
    '''Plots the regularization paths for LassoNet and gLASSO.'''

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, len(datapreds), figsize=figsize)

    for j, datapred in enumerate(datapreds):
        dataset, pred = datapred.split('/')

        for seed in [1]:
            reg_path = osp.join(selection_dir, f'{fs_method}/{dataset}/{pred}_reg_{seed}.pkl')
            reg = pickle.load(open(reg_path, 'rb'))

            n_selected = reg['n_selected']
            aurocs = reg['aurocs']
            lambda_ = reg['lambda_']
            n_selected_np = np.array(n_selected)
            selected_idx = np.argmin(
                np.abs(n_selected_np - N_TO_SELECT[f'{dataset}/{pred}/0.3'])
            )

            # AUROC vs. Number of Features
            axes[j].plot(
                n_selected, aurocs, marker='o', mfc='none', markersize=5, #label=f'Seed={seed}'
            )
            axes[j].scatter(
                n_selected[selected_idx], 
                aurocs[selected_idx], 
                marker='*',
                s=350,
                label='Selected Regularization Strength'
            )
            axes[j].set_title(DATAPRED_TO_PLOTNAME[datapred], fontsize=25)
            axes[j].axvline(
                x=N_TO_SELECT[f'{dataset}/{pred}/0.3'], color='black', linestyle='dashed', label='No. Features Selected by LLM-Score'
            )
            axes[j].set_xlabel('No. Features Selected', fontsize=25)
            axes[j].set_ylabel('AUROC', fontsize=25)
            axes[j].tick_params(axis='both', labelsize=20)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=5,
        bbox_transform=fig.transFigure,
        fontsize=25
    )
    
    if save_png:
        plot_path = osp.join(selection_dir, f'{fs_method}/{fs_method}_reg_path.png')
        plt.savefig(plot_path, bbox_inches='tight')

    elif save_pdf:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman"
        })

        plot_path = osp.join(selection_dir, f'{fs_method}/reg_path.pdf')
        plt.savefig(plot_path, bbox_inches='tight', backend='pgf', dpi=1600)

    else:
        plt.show()