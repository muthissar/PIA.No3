from functools import partial
from typing import List
from matplotlib import colors
import partitura as pt
import partitura.musicanalysis as musan
from ic.data import DataPiece, Piece
import importlib
from CIA.getters import get_dataloader_generator
from CIA.data_processors.data_processor import DataProcessor
from CIA.getters import get_data_processor
from pathlib import Path
from pretty_midi import PrettyMIDI
import json
import re
from ic.beam_search.io import ICRes
import torch
import os
import pretty_midi
from CIA.dataloaders.piano_dataloader import PianoDataloaderGenerator
from scipy.stats import pearsonr, spearmanr, kendalltau

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from ic.curves import Interpolator
from jsonargparse import ActionConfigFile, ArgumentParser

from ic.curves.weight import Hann, Weight
import numpy as np
from tqdm.contrib.concurrent import process_map,thread_map
import tqdm

from ic.eval_ import get_batik_matched_note_array, compute_windowed_metric
from ic.eval_ import get_all_token_ics_batik
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import appdirs
batik_dir = '/share/hel/home/mathias/datasets/batik_plays_mozart'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



recompute_match = True
n_workers = 36
piece_dict =  get_all_token_ics_batik()
def tqdm_map(f, *iterables, **kwargs):
    try:
        extra_kwargs = {'total' : len(iterables[0])}
    except Exception as e:
        print(e)
        extra_kwargs = {}
    return tqdm.tqdm(ThreadPoolExecutor.map(f, *iterables, **kwargs), **extra_kwargs)
# [partial(compute_windowed_metric,piece_dict=piece_dict )(x) ]
# process_map(partial(get_batik_matched_note_array, recompute_cache=recompute_match), piece_dict.keys(), max_workers=n_workers)
# res = tqdm.tqdm(map(partial(compute_windowed_metric, piece_dict=piece_dict), piece_dict.keys()), total=len(piece_dict))
# res = process_map(partial(compute_windowed_metric, piece_dict=piece_dict), piece_dict.keys(), max_workers=n_workers)
# timeout = 5 * 60  # 5 minutes
# timeout = 5
# with ThreadPoolExecutor(max_workers=n_workers) as executor:
#     futures = {executor.submit(compute_windowed_metric, piece, piece_dict): piece for piece in piece_dict.keys()}
#     results = []
#     # # for future in as_completed(futures, timeout=timeout):
#     # for future in futures:
#     #     piece = futures[future]
#     #     # results.append((piece, future.result()))
#     #     try:
#     #         results.append((piece, future.result(timeout=timeout)))
#     #     except TimeoutError as e:
#     #         print(f"Computation timed out {e}")
#         # for future in as_completed(futures, timeout=timeout):
#     for future in as_completed(futures, timeout=timeout):
#         piece = futures[future]
#         try:
#             results.append((piece, future.result()))
#         except TimeoutError as e:
#             print(f"Computation timed out {e}")
# get_batik_matched_note_array('kv331_3', recompute_cache=True)
results = []
scores = []
repetead_bars  : List[List]= []

matched = False
# pieces = [str(s.name)[:7] for s in  (Path(appdirs.user_cache_dir(), 'pia_eval').glob('*.pt'))]
pieces = sorted([str(s.name)[:7] for s in  (Path('/share/hel/home/mathias/datasets/batik_plays_mozart/scores').glob('*.musicxml'))])
# type_onsets = 
with tqdm.tqdm(pieces) as it:
    for piece in it:
        cache_file = Path(appdirs.user_cache_dir(), 'pia_eval', f'{piece}_note_array.pt')
        repetead_bars.append([])
        # if not cache_file.exists():
        #     continue
        # try:
        results.append(compute_windowed_metric(piece, piece_dict, matched=matched))
        part  = pt.load_score(Path(batik_dir, 'scores', f'{piece}.musicxml')).parts[0]
        # repeat_mes = part.measure_number_map([(r.start.t, r.end.t) for r in part.repeats])
        unfolded_measure_idx = 0
        snote_array = part.note_array()
        for r in part.repeats:
            start, end = part.measure_number_map((r.start.t, r.end.t)) - 1 
            n_repeats = end - start
            repetead_bars[-1].append(np.arange(n_repeats) + end + unfolded_measure_idx)
            unfolded_measure_idx += n_repeats
            
            # ax[0].plot(ic_success_summed, '-o', linewidth=lw, markersize=ms)
            # ax[0].plot(success_tt_cd, '-o', linewidth=lw, markersize=ms)
            # ax[0].plot(ic_success_summed*success_tt_cd, '-o', linewidth=lw, markersize=ms)

            # # Set major ticks at multiples of 16
            # ax[0].xaxis.set_major_locator(ticker.MultipleLocator(4))
            # ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())

            # # Set minor ticks at multiples of 8
            # ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))

            # # Set tick parameters
            # ax[0].tick_params(which='major', length=10, width=2)
            # ax[0].tick_params(which='minor', length=5, width=.5)

            # ax[0].grid(True, 'both')
        # part = pt.score.unfold_part_alignment(part=part, alignment=alignment ) #score.part.unfold_part_alignme
        # pt.save_score_midi(part, Path(out_folder, f'midi/{piece}.midi'))
        # except Exception as e:
        #     print(piece, " does not work. failed with ", str(e))
# torch.save((results, repetead_bars), 'raw_results.pt')
torch.save((results, repetead_bars), 'raw_results_score_positions_all.pt')
ics_summed_notes, win_ics_notes, notes_window_counts,  ics_summed_tt, win_ics_tt, tt = zip(*results)
succes_mask = np.array([True if x is not None and not np.isnan(x).any() else False for x in ics_summed_tt])
only_take_n_first = 8
bar_slice = slice(0, only_take_n_first)
# bar_slice = slice(0, None)
arr = np.array([x[bar_slice] for x,b in zip(ics_summed_tt, succes_mask) if b])
# arr = arr / np.max(arr, axis=1)
ic_success_summed = np.concatenate(arr)
success_tt = np.concatenate([x[bar_slice] for x,b in zip(tt,succes_mask) if b])
success_tt_cd = success_tt['cloud_diameter']
np.save('corrs.npy', np.stack((ic_success_summed, success_tt_cd), axis=0))
# ic_last_summed = np.exp(-ic_last_summed)
ic_success_summed_std = (ic_success_summed - np.mean(ic_success_summed))/np.std(ic_success_summed)
success_tt_cd_std = (success_tt_cd - np.mean(success_tt_cd))/np.std(success_tt_cd)
# tt = np.concatenate(tt)
# ics_summed_tt = np.concatenate(ics_summed_tt)
# print(pearsonr(ics_summed_tt, tt['cloud_diameter']), pearsonr(ic_last_summed, last_tt['cloud_diameter']))
# TODO: judging on the plot of the correlation, a more approriate correlation would be the polyserial correlation 
# however, from the histogram, it does not look like that there is a 
# https://search.r-project.org/CRAN/refmans/polycor/html/polyserial.html
print(pearsonr(ic_success_summed, success_tt_cd))
spearmanr(ic_success_summed, success_tt_cd)
kendalltau(ic_success_summed, success_tt_cd)

# ic_last_summed_scl = np.array(ic_last_summed)/np.max(ic_last_summed)
fig, ax = plt.subplots(nrows=2, ncols=1) #, figsize=(20,9))
ax : List[plt.Axes] = ax
lw = 1.0
ms = 2.0
h = ax[0].hist2d(ic_success_summed_std, success_tt_cd_std, range=[[-1.5, 2],[-2, 2.5]],bins=50, cmap='inferno', norm=colors.LogNorm())
plt.colorbar(h[3], ax=ax)
ax[0].set_xlabel('ic')
ax[0].set_ylabel('cd')

ax[1].scatter(ic_success_summed_std, success_tt_cd_std, s=ms)
ax[1].set_xlabel('ic')
ax[1].set_ylabel('cd')

out_folder = "out/quant"
# plt.legend(['IC', 'TT','IC*TT'])
plt.savefig(f'{out_folder}/figs/aggregated.pdf')
matched_array = get_batik_matched_note_array(piece, recompute_cache=False)
onset_score = matched_array['onset_sec']
onset_ic = piece_dict[piece]['tok_times'].min(1)[0]
index2value, value2index = torch.load('mapper_dicts.pt')
pitch_ic = [index2value['pitch'][x[0].item()] for x in piece_dict[piece]['toks']]
pitch_score = matched_array['pitch']
score = pt.load_score(Path(batik_dir, 'scores', f'{piece}.musicxml'))
# performance, alignment = pt.load_match(Path(batik_dir, 'match', f'{piece}.match'), first_note_at_zero=True)
# part = score.parts[0]


res_pieces, res = zip(*results)
torch.save(res, 'quant_metrics.pt')
ics_summed = []
tc = [[],[],[]]
for r in res:
    ics_summed.extend(r[0])
    for i, tc_ in enumerate(tc,1):
        tc_.extend(r[i])
corrs = [pearsonr(ics_summed, tc_) for tc_ in tc]
torch.save(corrs, 'corrs.pt')
