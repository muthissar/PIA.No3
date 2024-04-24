from pathlib import Path
from jsonargparse import ArgumentParser
from matplotlib import pyplot as plt
import pandas as pd
import pretty_midi
import torch
import tqdm
from ic.eval.rhythm import Metrics
from ic.eval.rhythm.util import get_pattern
from ic.eval_ import get_all_token_ics_batik, get_batik_matched_note_array_2
from ic.curves import Interpolator
from ic.curves.weight import Hann
import appdirs
import builtins
import warnings
import slugify
from scipy.stats import pearsonr, norm #, spearmanr, kendalltau
import math
def r_to_z(r):
    return math.log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    e = math.exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, alpha, n):
    z = r_to_z(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))
def compute():    
    print_old = builtins.print
    builtins.print = lambda *args, **kwargs : None

    import partitura as pt
    import numpy as np
    complexities = {}
    piece_dict =  get_all_token_ics_batik()
    # for p in tqdm.tqdm(list(Path('/share/hel/home/mathias/datasets/batik_plays_mozart/scores/').glob('*.musicxml'))):
    for cached in tqdm.tqdm(list(Path(appdirs.user_cache_dir(), 'pia_eval_2').glob(f'*_snote_array.pt')), desc='Computing pr. piece complexities'):
        piece = cached.stem[:7]
        batik_dir = '/share/hel/home/mathias/datasets/batik_plays_mozart'
        p = Path(batik_dir,'scores',f'{piece}.musicxml')
        cw = [0.9099181, 0, 0, 0.68306011]
        weight = Hann(
            window_size=2.0,
            channel_weight=cw
        )
        interpolator = Interpolator(
            # NOTE: 
            metric_times = [piece_dict[piece]['tok_times']],
            metric = [piece_dict[piece]['tok_ics']],
            weight_fn = weight
        )
        matched_parray, matched_sarray = get_batik_matched_note_array_2(piece, recompute_cache=False)
        staffs = matched_sarray['staff'].filled()
        metpos = np.lib.recfunctions.structured_to_unstructured(matched_sarray[['rel_onset_div', 'tot_measure_div']].filled())
        # metpos = metpos.astype((metpos.dtype[0], len(metpos.dtype.names)))
        measures = matched_sarray['nmn'].filled()
        unique_measures = np.unique(measures)

        complexities[p.stem] = {
                'rhythmic_complexity':{},
                'iic': []
        }
        take_n_measures = 99999
        # for meas_n in unique_measures[slice(0, take_n_measures)]:
        # NOTE: all scores start at t=0.0
        measure_start = 0.0
        for i in range(min(take_n_measures, len(unique_measures))):
            staff_colls = [[*np.unique(staffs)[:,None].tolist()], [np.unique(staffs).tolist()]]
            compl_staff_coll  = {}
            for staff_coll in staff_colls:
                compl_staff = compl_staff_coll.setdefault(str(staff_coll), {})
                for staff in staff_coll:
                    meas_n = unique_measures[i]
                    measure =  metpos[(measures == meas_n) & np.isin(staffs, staff )]
                    if len(measure) == 0:
                        continue
                    # TODO: figure out if the lengths needs to be normalized
                    assert len(set(measure[:, 1])) == 1, "Not implimented if the denom is changing during a measure."
                    length = measure[0,1]
                    onsets_indeces = np.unique(measure[:, 0])
                    # gcd = np.gcd.reduce(onsets_indeces.tolist() + [length]) 
                    gcd = 1
                    length = length // gcd
                    onsets_indeces = onsets_indeces // gcd
                    
                    # Instantiate a metric class
                    metrics = Metrics(length, onsets_indeces)
                    compl_metrics = {
                        'lhl': metrics.getLonguetHigginsLeeComplexity,
                        'smith': metrics.getSmithHoningComplexity,
                        'iecl': lambda : metrics.getInformationEntropyComplexity()[1],
                        'iecg': lambda : metrics.getInformationEntropyComplexity()[0],
                        'ceps':  metrics.getCEPSComplexity,
                        'keith': metrics.getKeithComplexity,

                    }
                    for func_name, func in compl_metrics.items():
                        rc = compl_staff.setdefault(func_name, [])
                        rc.append(func())
            for staff_coll, compl_staff in compl_staff_coll.items():
                for metric, metric_vals in compl_staff.items():
                    # NOTE: we sum together, rather than mean the contributions of the staffs such that a staff with no notes,
                    # counts less than two which are  
                    complexities[p.stem]['rhythmic_complexity'].setdefault(staff_coll,{}).setdefault(metric, []).append(np.sum(metric_vals))

            # NOTE: performance
            notes_in_current_measure = matched_parray[np.isin(matched_parray['id'],matched_parray['id'][measures == unique_measures[i]])]
            measure_end_l = np.max(notes_in_current_measure['onset_sec'])
            if i < len(unique_measures)-1:
                notes_in_next_measure = matched_parray[np.isin(matched_parray['id'],matched_parray['id'][measures == unique_measures[i+1]])]
                measure_end_r = np.min(notes_in_next_measure['onset_sec'])
            else:
                notes_in_current_measure
                measure_end_r = measure_end_l
            measure_end = np.mean([measure_end_l,measure_end_r])
            iic = interpolator([torch.arange(measure_start, measure_end, 0.1)])
            complexities[p.stem]['iic'].append(iic[0])
            measure_start = measure_end
    builtins.print = print_old
    corrs = []
    for i in tqdm.tqdm(range(1,max(len(piece['iic']) for piece in complexities.values())+1), desc='Aggregating pr. piece results'):
        rhythmic_complexities = {}
        iic = dict(
            pitch = [],
            timeshift = [],
            both = [],
        )
        for piece in complexities.values():
            # for rhythmic_type in piece['rhythmic_complexity']:
            for staff_coll_name, staff_coll in piece['rhythmic_complexity'].items():
                for rhythmic_type_name in staff_coll:
                    rhythmic_complexities.setdefault(staff_coll_name, {}).setdefault(rhythmic_type_name, []).extend(staff_coll[rhythmic_type_name][:i])
            iic['pitch'].extend([iic[:, 0].mean().item() for iic in piece['iic'][:i]])
            iic['timeshift'].extend([iic[:, 3].mean().item() for iic in piece['iic'][:i]])
            iic['both'].extend([iic.sum(-1).mean().item() for iic in piece['iic'][:i]])
        for staff_coll_name, staff_coll in rhythmic_complexities.items():
            for rhythmic_type_name, rhythmic_type in staff_coll.items():
                for iic_type in iic:
                    corr, p = pearsonr(iic[iic_type], rhythmic_type)
                    n_measures = len(iic[iic_type])
                    conf_l, conf_h = r_confidence_interval(r=corr, alpha=0.05,n=n_measures)
                    corrs.append({
                        'complexity_type': rhythmic_type_name,
                        'iic_type': iic_type,
                        # '$\rho$': corr,
                        'r': corr,
                        'r_l': conf_l,
                        'r_h': conf_h,
                        'r_n': n_measures,
                        'p': p,
                        # 'First $n$ measures.': i,
                        'n': i,
                        'staff_coll': staff_coll_name,
                    })
    corrs = pd.DataFrame(corrs)
    corrs.to_hdf('out/quant/rhythmic_corrs.h5',key='corr')
def plot():
    corrs = pd.read_hdf('out/quant/rhythmic_corrs.h5',key='corr')
    comp_formats = {
        'smith': ('orange', 'H'),
        'lhl': ('red', 'D'),
        'iecl': ('blue', 'o'),
        'iecg': ('green', 's'),
        'ceps': ('purple', '^'),
        'keith': ('black', 'v'),
    }
    for staff_group_name, staff_group in corrs[~corrs.r.isna()].groupby('staff_coll'):
        grouped = staff_group.groupby(['complexity_type', 'iic_type'])
        fig, ax = plt.subplots(figsize=(10, 5))
        cs = []
        styles = []
        labels = []

        for (complexity_type, iic_type), group in grouped:
        # for (iic_type, complexity_type) in [('pitch',)]:
            # group = grouped.get_group((iic_type, complexity_type))
            if iic_type == 'pitch':
                style='--'
            elif iic_type == 'timeshift':
                style=':'
            elif iic_type == 'both':
                style='-'
                continue
            else:
                raise NotImplementedError
            c, style_ = comp_formats[complexity_type]
            style += style_
            cs.append(c)
            styles.append(style)
            labels.append(f'${complexity_type}_{{{iic_type}}}$')
            # group.plot(x='First $n$ measures.', y='$\rho$', ax=ax, style=style, c=c, legend=f'${complexity_type}_{{{iic_type}}}$', markevery=.1)
            group.plot(x='n', y='r', ax=ax, style=style, c=c, legend=f'${complexity_type}_{{{iic_type}}}$', markevery=.1,
            # ylabel='$\\rho$', xlabel='First $n$ measures.'
            )
            # ax.fill_between(group['First $n$ measures.'], group['r_l'], group['r_h'], color=c, alpha=0.1)
            ax.fill_between(group['n'], group['r_l'], group['r_h'], color=c, alpha=0.1)
        # grouped.plot(x='n', y='corr', ax=ax, style=styles)
        ax.set_xlabel('First $n$ measures.')
        ax.set_ylabel('$\\rho$')
        ax.set_xlim((1,382))
        ax.set_xscale('log')
        ax.legend(labels, loc='upper right')
        staff_name_mapper = {'[[1, 2]]': 'staff_onset_union', '[[1], [2]]': 'staff_meaned'}
        ax.set_title(f'Correlation between $IIC$ and rhythmic complexity using {staff_name_mapper[staff_group_name]}.')
        fig.savefig(f'out/quant/figs/rhythmic_complexity_{staff_name_mapper[staff_group_name]}.pdf')
if __name__ == '__main__':
    parser = ArgumentParser()
    commands = parser.add_subcommands()
    compute_parser = ArgumentParser() 
    plot_parser = ArgumentParser()
    commands.add_subcommand("compute", compute_parser)
    commands.add_subcommand("plot", plot_parser)
    args = parser.parse_args()
    if args.subcommand == 'compute':
        compute()
    elif args.subcommand == 'plot':
        plot()
    else:
        compute()
        plot()
