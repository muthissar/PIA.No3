from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import pretty_midi
import torch
import tqdm
from ic.eval.rhythm import Metrics
from ic.eval.rhythm.util import get_pattern
from ic.eval_ import get_all_token_ics_batik, get_batik_matched_note_array
from ic.curves import Interpolator
from ic.curves.weight import Hann
import appdirs
import builtins
import warnings
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
print_old = builtins.print
builtins.print = lambda *args, **kwargs : None
# from MIDIFunctions import plot_beats



# Datasets

# Povel & Essens patterns
# pm_PovEss = pretty_midi.PrettyMIDI('./data/Povel&Essens 1985/Povel&Essens - 1.mid')

# Essens patterns
# pm_Ess = pretty_midi.PrettyMIDI('./data/Essens 1995/Essens - 1.mid')

# Fitch & Rosenfeld patterns
# pm_FitRos = pretty_midi.PrettyMIDI('./data/Fitch&Rosenfeld 2007/Fitch&Rosenfeld - 15.mid')



# Select a single pattern for rhythmic complexity analysis
# pm = pm_FitRos
import partitura as pt
import numpy as np
# failed = {}

# if channel_weight == 'pitch':
#     cw[3] = 0.0
# elif channel_weight == 'timeshift':
#     cw[0] = 0.0
# elif channel_weight != 'both':
#     raise NotImplementedError
complexities = {}
piece_dict =  get_all_token_ics_batik()
# for p in tqdm.tqdm(list(Path('/share/hel/home/mathias/datasets/batik_plays_mozart/scores/').glob('*.musicxml'))):
for cached in tqdm.tqdm(list(Path(appdirs.user_cache_dir(), 'pia_eval').glob(f'*_note_array.pt'))):
    piece = cached.stem[:7]
    p = Path('/share/hel/home/mathias/datasets/batik_plays_mozart/scores',f'{piece}.musicxml')
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
    matched_array = get_batik_matched_note_array(piece, recompute_cache=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        score = pt.load_musicxml(str(p))
        part = score.parts[0]
        snote_array = part.note_array(
            # include_pitch_spelling=True,
            # include_key_signature=True,
            # include_metrical_position=True,
            include_staff=True,
        )
        measures = part.measure_number_map(snote_array['onset_div'])
        metpos = part.metrical_position_map(snote_array['onset_div'])
        staffs = snote_array['staff']
    # NOTE: map to quarters.
    # part.quarter_map(metpos) 
    unique_measures = np.unique(measures)

    complexities[p.stem] = {
            'rhythmic_complexity':{
                # 'smith': [],
                # 'lhl': [],
            },
            'iic': []
    }
    take_n_measures = 20
    # for meas_n in unique_measures[slice(0, take_n_measures)]:
    # NOTE: all scores start at t=0.0
    measure_start = 0.0
    for i in range(take_n_measures):
        compl_staff = {}
        for staff in np.unique(staffs):
            meas_n = unique_measures[i]
            measure =  metpos[(measures == meas_n) & (staffs == staff )]
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
            for func_name, func in {'lhl': metrics.getLonguetHigginsLeeComplexity, 'smith': metrics.getSmithHoningComplexity}.items():
                rc = compl_staff.setdefault(func_name, [])
                rc.append(func())
        for metric, value in compl_staff.items():
            complexities[p.stem]['rhythmic_complexity'].setdefault(metric, []).append(np.mean(value))

        # NOTE: performance
        notes_in_current_measure = matched_array[np.isin(matched_array['id'],snote_array['id'][measures == meas_n])]
        measure_end_l = np.max(notes_in_current_measure['onset_sec'])
        if i < len(unique_measures)-1:
            notes_in_next_measure = matched_array[np.isin(matched_array['id'],snote_array['id'][measures == unique_measures[i+1]])]
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
for i in range(1,take_n_measures):
    rhytmic_complexities = {}
    iic = dict(
        pitch = [],
        timeshift = [],
        both = [],
    )
    for value in complexities.values():
        for rhythmic_type in value['rhythmic_complexity']:
            rhytmic_complexities.setdefault(rhythmic_type, [])
            rhytmic_complexities[rhythmic_type].extend(value['rhythmic_complexity'][rhythmic_type][:i])
        iic['pitch'].extend([iic[:, 0].mean().item() for iic in value['iic'][:i]])
        iic['timeshift'].extend([iic[:, 3].mean().item() for iic in value['iic'][:i]])
        iic['both'].extend([iic.sum(-1).mean().item() for iic in value['iic'][:i]])
    for rhythmic_type in rhytmic_complexities:
        for iic_type in iic:
            corr, p = pearsonr(iic[iic_type], rhytmic_complexities[rhythmic_type])
            conf_l, conf_h = r_confidence_interval(r=corr, alpha=0.05,n=len(iic[iic_type]))
            corrs.append({
                'complexity_type': rhythmic_type,
                'iic_type': iic_type,
                '$\rho$': corr,
                'r_l': conf_l,
                'r_h': conf_h,
                'p': p,
                'First $n$ measures.': i
            })
corrs = pd.DataFrame(corrs)
fig, ax = plt.subplots()
cs = []
styles = []
labels = []
grouped = corrs.groupby(['complexity_type', 'iic_type'])
for (complexity_type, iic_type), group in grouped:
# for (iic_type, complexity_type) in [('pitch',)]:
    # group = grouped.get_group((iic_type, complexity_type))
    if iic_type == 'pitch':
        style='--'
    elif iic_type == 'timeshift':
        style=':'
    elif iic_type == 'both':
        style='-'
    else:
        raise NotImplementedError
    if complexity_type == 'smith':
        c = 'orange'
        style += 'H'
    elif complexity_type == 'lhl':
        c = 'red'
        style += 'D'
    else:
        raise NotImplementedError
    cs.append(c)
    styles.append(style)
    labels.append(f'${complexity_type}_{{{iic_type}}}$')
    group.plot(x='First $n$ measures.', y='$\rho$', ax=ax, style=style, c=c, legend=f'${complexity_type}_{{{iic_type}}}$')
    # ax.fill_between(group['First $n$ measures.'], group['r_l'], group['r_h'], color=c, alpha=0.1)
# grouped.plot(x='n', y='corr', ax=ax, style=styles)
ax.legend(labels)
ax.set_title('Correlation between $IIC$ and rhythmic complexity.')
fig.savefig('out/quant/figs/rhythmic_complexity.pdf')