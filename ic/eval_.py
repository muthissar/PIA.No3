from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import pandas as pd
from ic.beam_search.io import ICRes
from ic.config import Config
#  present_df


import numpy as np
import torch
import tqdm


from typing import List
from warnings import warn

import partitura as pt
import appdirs
import partitura.score as spt
import re
from partitura.utils.globals import ALT_TO_INT, INT_TO_ALT
from partitura.utils.music import estimate_symbolic_duration, transpose_note

from ic.curves.weight import Hann
from ic.curves import Interpolator

def eval_(configs : List[Config], out_file: str):
    import pandas as pd
    # Meta data for each experiment
    exps = []
    params = []
    hash_ = []
    pieces = []
    samples = []

    # observations related to the interpolation
    int_times = []
    int_ic_devs = []
    int_ids = []

    # observations related to the tokens
    tok_times = []
    tok_ic = [[], [], [], []]
    tok_ids = []

    # observations related to the piece
    piece_pad_before = []
    piece_pad_after = []
    piece_n_inpaint = []
    piece_ids = []


    # What I should do instead, is simply creating a muliti index (used for both datasets) and have a note table and an int table
    # The multiindex should then in the latter apper multiple times for observations
    id_ = 0
    for c in tqdm.tqdm(configs, desc='Experiment', leave=False):
        for piece_dir in tqdm.tqdm([f for f in c.out.glob('*') if f.is_dir()], desc='Piece', leave=False):
            temp_file = piece_dir.joinpath('temp', 'ic.pt')
            if temp_file.exists():
                tem = ICRes.load(p=str(temp_file))
                # win_steps = 0.5
                # # win_length = 0.5
                # total_len  = 9.7
                # wins = np.arange(win_steps, total_len, win_steps)
                # ids = [[] for ids in range(len(wins))]
                # for i in range(1,len(wins)):
                #     warn('For now the time of the note is found by meaning the time of the tokens, this might change in future.')
                #     timepoints = tem.timepoints.mean(axis=-1)
                #     for enumed, t in enumerate(timepoints):
                #         if t > wins[i] - win_length and t < wins[i]:
                #             ids[i].append(enumed)



                # data_processor PAD_SYMBOL
                # TODO: hard_coded
                pad_symb = torch.tensor([88, 128, 104, 105])[None]
                before = tem.tok[:256]
                after = tem.tok[257:513]
                middle = tem.tok[514:tem.decoding_end]

                piece_ids.append(id_)
                piece_pad_before.append((before == pad_symb).all(-1).sum().item())
                piece_pad_after.append((after == pad_symb).all(-1).sum().item())
                piece_n_inpaint.append(middle.shape[0])
                # warn('Add the refrence, below is not working since it will also be used when computing other means.')
                # NOTE: this adds the  same reference file for each experiments...
                tok_times.extend(tem.timepoints.numpy())
                # tok_ic.extend(tem.ic_tok.numpy().mean(axis=-1))
                for ic, feat in zip(tem.ic_tok.numpy().T, tok_ic):
                    feat.extend(ic)
                tok_ids.extend(len(tem.ic_tok)*[id_])
                exps.append(str(c.experiment))
                hash_.append(str(c.out))
                params.append("ref")
                pieces.append(piece_dir.name)
                samples.append("ref")
                id_+=1
                for sample in piece_dir.rglob('*/ic.pt'):
                    # TODO: rename
                    tem = ICRes.load(p=str(piece_dir.joinpath('match', 'ic.pt')))
                    if sample.parent.name not in ['temp', 'match']:
                        gen = ICRes.load(p=sample)
                        before = gen.tok[:256]
                        after = gen.tok[257:513]
                        middle = gen.tok[514:gen.decoding_end]
                        piece_ids.append(id_)
                        piece_pad_before.append((before == pad_symb).all(-1).sum().item())
                        piece_pad_after.append((after == pad_symb).all(-1).sum().item())
                        piece_n_inpaint.append(middle.shape[0])

                        ic_dev = tem.ic_int.sum(dim=-1) - gen.ic_int.sum(dim=-1)
                        int_ic_devs.extend(ic_dev.numpy())
                        int_times.extend(gen.timepoints_int.numpy())
                        int_ids.extend(len(ic_dev)*[id_])
                        tok_times.extend(gen.timepoints.numpy())
                        # tok_yic.extend(gen.ic_tok.numpy().mean(axis=-1))
                        for ic, feat in zip(gen.ic_tok.numpy().T, tok_ic):
                            feat.extend(ic)
                        tok_ids.extend(len(gen.ic_tok)*[id_])
                        # exps.append(str(c.experiment))
                        exps.append(json.dumps(asdict(c.experiment)))
                        hash_.append(str(c.out))
                        # exps.append(c.experiment)
                        # TODO: make more general
                        # params.append(str(c.sampling_config))
                        params.append(json.dumps(asdict(c.sampling_config)))
                        # params.append(c.sampling_config)
                        pieces.append(piece_dir.name)
                        samples.append(sample.parent.name)
                        id_+=1
    # index = pd.MultiIndex.from_arrays([pd.Categorical(exps), pd.Categorical(params), pd.Categorical(pieces), pd.Categorical(samples)], names=('exps', 'params', 'piece', 'sample'))
    ex = pd.DataFrame({
        # NOTE: categorical does not work with h5
        # 'exps': pd.Categorical(exps),
        # 'params': pd.Categorical(params),
        # 'piece': pd.Categorical(pieces),
        # 'sample': pd.Categorical(samples)
        'exps': pd.Series(exps,dtype=str),
        'params': pd.Series(params, dtype=str),
        'hash': pd.Series(hash_, dtype=str),
        'piece': pd.Series(pieces,dtype=str),
        'sample': pd.Series(samples, dtype=str)
    })
    # TODO: seems uncesserary but don't know how to do groupby with the multiindex alone
    # index_df = pd.DataFrame(np.arange(len(index)), index=index)
    # index.groupby()
    int_df = pd.DataFrame({
        'ids': int_ids,
        'time' : pd.Series(int_times, dtype=np.float64),
        'ic_dev' : pd.Series(int_ic_devs, dtype=np.float64)
    })
    
    ic_pitch = tok_ic[0]
    ic_vel = tok_ic[1]
    ic_dur = tok_ic[2]
    ic_shift = tok_ic[3]
    if c.experiment.onset_on_next_note:
        changes = np.where(np.diff(tok_ids) > 0)[0]
        tok_times_arr = np.array(tok_times)
        times_split = np.split(tok_times_arr, changes+1)
        tok_times_ = []
        # add extra points id per id
        for t in times_split:
            tok_times_.extend(list(t[:,0 ]) + [t[-1,-1]])
        tok_ids_ = []
        for t in np.split(tok_ids, changes+1):
            tok_ids_.extend(np.insert(t, 0, t[0]))
        ic_pitch_ = []
        for t in np.split(ic_pitch, changes+1):
            ic_pitch_.extend(np.insert(t, -1, np.nan))
        ic_vel_ = []
        for t in np.split(ic_vel, changes+1):
            ic_vel_.extend(np.insert(t, -1, np.nan))
        ic_dur_ = []
        for t in np.split(ic_dur, changes+1):
            ic_dur_.extend(np.insert(t, -1, np.nan))
        ic_shift_ = []
        for t in np.split(ic_shift, changes+1):
            ic_shift_.extend(np.insert(t, 0, np.nan))
        tok_ic_ = [ic_pitch_, ic_vel_, ic_dur_, ic_shift_]
    else:
        tok_ids_ = tok_ids
        tok_times_ = np.mean(tok_times, axis=-1)
        tok_ic_ = tok_ic
        ic_pitch_ = ic_pitch
        ic_vel_ = ic_vel
        ic_dur_ = ic_dur
        ic_shift_ = ic_shift
        

    
    tok_df = pd.DataFrame({
        'ids': tok_ids_,
        'time' : pd.Series(tok_times_, dtype=np.float64),
        'ic_pitch' : pd.Series(ic_pitch_, dtype=np.float64),
        'ic_vel' : pd.Series(ic_vel_, dtype=np.float64),
        'ic_dur' : pd.Series(ic_dur_, dtype=np.float64),
        'ic_shift' : pd.Series(ic_shift_, dtype=np.float64),
        'ic_mean': pd.Series(np.nanmean(tok_ic_, axis=0), dtype=np.float64)
    })

    piece_df = pd.DataFrame({
        'ids': piece_ids,
        'pad_before': pd.Series(piece_pad_before, dtype=np.int),
        'pad_after': pd.Series(piece_pad_after, dtype=np.int),
        'n_inpaint': pd.Series(piece_n_inpaint, dtype=np.int),
    })

    # out_file = 'out/results/result.h5'
    ex.to_hdf(out_file, key='ex')
    int_df.to_hdf(out_file, key='int_df')
    tok_df.to_hdf(out_file, key='tok_df')
    piece_df.to_hdf(out_file, key='piece_df')
    present_df(ex, int_df, tok_df)


def present_df(ex, int_df, tok_df):
    merged = ex.merge(int_df, left_index=True, right_on="ids",how='outer')
    res_int = merged[merged['params'] != 'ref'].groupby(['exps', 'params']).agg({'ic_dev': ['mean', ('abs_mean', lambda x: x.abs().mean()), 'std', 'count']})
    # def sort_fun(x):
    #     def sort_fun_(x):
    #         try:
    #             return int(x)
    #         except ValueError: 
    #             return -1
    #     sorted(x, key=sort_fun_)
    # res_int = res_int.sort_index(level=['params'], key=sort_fun)
    print(res_int.round(3))

    # merged = ex.merge(tok_df, left_index=True, right_on="ids",how='outer').groupby(['exps', 'params'])
    # res_not = merged.agg({'ics': ['mean', 'count']})
    merged = ex.merge(tok_df, left_index=True, right_on="ids",how='outer')
    cond = (merged['sample'] == 'ref') ^ (merged['params'] != 'ref')
    res_not = merged.groupby([merged['exps'], merged['params'], cond]).agg({'ic_pitch': ['count', 'mean', 'min', 'max'], 'ic_vel': ['mean', 'min', 'max'], 'ic_dur': ['mean', 'min', 'max'], 'ic_shift': ['mean', 'min', 'max'], 'ic_mean': ['mean']})

    print(res_not.round(3))

def get_batik_matched_note_array(piece_name: str, batik_dir: str='/share/hel/home/mathias/datasets/batik_plays_mozart', recompute_cache : bool =False) -> np.ndarray:
    cache_file = Path(appdirs.user_cache_dir(), 'pia_eval', f'{piece_name}_note_array.pt')
    if cache_file.exists() and not recompute_cache:
        matched_array = torch.load(str(cache_file))
    else:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # harmony = pd.read_csv(Path(batik_dir, 'score_parts_annotated', f'{piece_name}_spart_harmony.csv'), index_col='id')
        score = pt.load_score(Path(batik_dir, 'scores', f'{piece_name}.musicxml'))
        performance, alignment = pt.load_match(Path(batik_dir, 'match', f'{piece_name}.match'), first_note_at_zero=True)
        part = score.parts[0]
        data = pd.read_csv(Path(batik_dir, 'score_parts_annotated', f'{piece_name}_spart_harmony.csv'))
        data = data.assign(
            ks_fifths = pd.Series(dtype=np.int32),
            ks_mode = pd.Series(dtype=np.int32)
        )
        # NOTE: this code is calling partitura PR #345, but does not work, since the roman numerals are not copied to the note_array 
        # with tempfile.NamedTemporaryFile(suffix='.tsv') as temp:
        #     data = data.rename(columns={'onset_quarter': 'quarterbeats', 'duration_quarter': 'duration_qb'})
        #     data['cadence'] = float('nan')
        #     data['phraseend'] = float('nan')
        #     data.to_csv(temp.name, sep='\t', index=False)
        #     pt.io.importdcml.read_harmony_tsv(beat_tsv_path=temp.name, part=part)

        # NOTE: this code is adapted from partitura.io.importdcml.read_harmony_tsv in PR #345, 
        is_na_roman = data["chord"].isna()
        for idx, row in data[~is_na_roman].iterrows():
            # row["chord_type"] contains the quality of the chord but it is encoded differently than for other formats
            # and datasets. For example, a minor chord is encoded as "m" instead of "min" or "minor"
            # Therefore we do not add the quality to the RomanNumeral object. Then it is extracted from the text.
            # Local key is in relation to the global key.
            # TODO: in kv331_3_spart_harmony.csv/#L760 the local key is #vi, which is not supported by Roman2Interval_Min
            if row["globalkey"].islower():
                transposition_interval = spt.Roman2Interval_Min[row["localkey"]]
            else:
                transposition_interval = spt.Roman2Interval_Maj[row["localkey"]]

            key_step = re.search(r"[a-gA-G]", row["globalkey"]).group(0)
            key_alter = re.search(r"[#b]", row["globalkey"]).group(0) if re.search(r"[#b]", row["globalkey"]) else ""
            key_alter = key_alter.replace("b", "-")
            key_alter = ALT_TO_INT[key_alter]
            key_step, key_alter = transpose_note(key_step, key_alter, transposition_interval)
            local_key = key_step + INT_TO_ALT[key_alter]
            local_key = local_key.replace("-", "b")
            ks_fifths, ks_mode = pt.utils.key_name_to_fifths_mode(local_key)
            # NOTE: this line might be unnecessary
            ks_mode = 1 if ks_mode == 'major' else -1
            data.loc[idx, ['ks_fifths', 'ks_mode']] = [ks_fifths, ks_mode]
        data[['ks_fifths', 'ks_mode']] = data[['ks_fifths', 'ks_mode']].fillna(method='ffill')
        part = pt.score.unfold_part_alignment(part=part, alignment=alignment ) #score.part.unfold_part_alignme
        snote_array = part.note_array(include_pitch_spelling=True,include_key_signature=True)
        data.index = data.id
        snote_array['ks_fifths'] = data.loc[snote_array['id'], 'ks_fifths']
        snote_array['ks_mode'] = data.loc[snote_array['id'], 'ks_mode']
        measures = part.measure_number_map(snote_array['onset_div'])
        # NOTE: rewritem measure numbers (incorrect due to repeats)
        counter = 0
        for i, is_change  in enumerate(np.diff(measures, prepend=-1)):
            if is_change:
                counter+=1
            measures[i] = counter
        snote_array = np.lib.recfunctions.append_fields(snote_array, names=["mn"], data=[measures])
        # new_measures = np.array(new_measures)
        # names = ['chord_info']
        # data = pd.Series(np.zeros(len(snote_array), dtype=bool),index=snote_array['id'])
            # data.loc[harmony.index] = ~harmony.chord.isna()
        # snote_array = np.lib.recfunctions.append_fields(snote_array, names=names, data=[data])
        pnote_array = performance.note_array()
        index_in_score_note_array, index_in_performance_notearray  = pt.musicanalysis.performance_codec.get_matched_notes(
                spart_note_array=snote_array, 
                ppart_note_array=pnote_array, 
                alignment=alignment,
            ).T
        append_names = ['step', 'alter', 'octave', 'ks_fifths', 'ks_mode', "mn"]
        append_data = [snote_array[index_in_score_note_array][name] for name in append_names] 
        matched_array = np.lib.recfunctions.append_fields(pnote_array[index_in_performance_notearray], names=append_names, data=append_data)
        torch.save(matched_array, str(cache_file))
    return matched_array

def get_all_token_ics_batik() -> dict:
    # get ic of all pieces in the batik ds. Asusme that the same shift, and number of notes was used for all examples.
    out_dir = Path('out/a6aa5f7b2689124d92900e130182eebe0394953f773fcfec9b9e9e1466e792da/samplingconfig-k-traces-128-temperature-1-0-n-poly-notes-none-dynamic-temperature-max-ic-50-0-top-p-0-0-top-k-0')
    pat = re.compile(r'^(?P<piece>.+)_start_(?P<start>-?\d+)_nodes_(?P<n_inpaint>-?\d+)_end_0$')
    decoding_start = 514
    min_notes_in_context = 256
    n_inpaint = 506
    hop_size = n_inpaint-min_notes_in_context

    def parse_path(p : Path) -> int:
        match = pat.match(p.name)
        return match.group('piece'), int(match.group('start')), int(match.group('n_inpaint'))
    paths = list(out_dir.glob('*_nodes_506_end_0'))
    piece_dict = {}
    parsed_paths = [parse_path(p) for p in  paths]
    # n_inpaint = parsed_paths[0][-1]
    for p, (piece, start, n_inpaint_loc) in zip(paths, parsed_paths):
        assert n_inpaint_loc == n_inpaint
        piece_dict.setdefault(piece, []).append((p,start))
    for key, val in piece_dict.items():
        val = sorted(val, key=lambda x: x[1])
        paths,_= zip(*val)
        piece_dict[key] = {'paths': paths}
    for piece_name in sorted(piece_dict.keys()):
        toks = []
        tok_times = []
        tok_ics = []
    # [x[0][decoding_start:decoding_start+n_inpaint]] + [x_[decoding_start+n_inpaint-hop_size:decoding_start+n_inpaint,0] for x_ in x[1:]]
        piece = piece_dict[piece_name]
        for i, p in enumerate(piece['paths']):
            res_path = p.joinpath('temp', 'ic.pt')
            if res_path.exists():
                res = ICRes.load(str(res_path))
                if i == 0:
                    ic_toks = res.ic_tok[:n_inpaint]
                    real_n_inpaint = res.ic_tok.shape[0]
                    toks.append(res.tok[decoding_start:decoding_start+real_n_inpaint])
                    tok_ics.append(res.ic_tok[:n_inpaint])
                    tok_times.append(res.timepoints[:n_inpaint])
                else:
                # NOTE: if there is end padding, it's dropped in the tok_ics
                    ic_toks = res.ic_tok[n_inpaint-hop_size:n_inpaint]
                    real_n_inpaint = res.ic_tok.shape[0]
                    toks.append(res.tok[decoding_start+n_inpaint-hop_size:decoding_start+real_n_inpaint])
                    tok_ics.append(ic_toks)
                    tok_times.append(res.timepoints[n_inpaint-hop_size:n_inpaint])
        piece['toks'] = torch.cat(toks, axis=0)
        piece['tok_ics'] = torch.cat(tok_ics, axis=0)
        piece['tok_times'] = torch.cat(tok_times, axis=0)
    return piece_dict


def compute_windowed_metric(piece_name : str, piece_dict: dict):
        piece = piece_dict[piece_name]
        matched_array = get_batik_matched_note_array(piece_name, recompute_cache=False)
        if not 'mn' in matched_array.dtype.names:
            raise ValueError('No matched notes')
        # assert pnote_array.shape[0] == piece['toks'].shape[0]
        weight = Hann(
                window_size=2.0,
                channel_weight=[0.9099181, 0, 0, 0.68306011]
                # channel_weight=[0.9099181, 0, 0, 0.0]
            )
        interpolator = Interpolator(
            # NOTE: 
            metric_times = [piece['tok_times']],
            metric = [piece['tok_ics']],
            weight_fn = weight
        )
        # NOTE: compute number of notes
        onsets = matched_array['onset_sec']
        notes_window_size = 4.0
        end = max(matched_array['onset_sec']) + notes_window_size
        notes_window_endpoints = np.arange(0, end, notes_window_size)
        notes_window_indices = np.digitize(onsets, notes_window_endpoints) -1
        notes_window_counts = np.bincount(notes_window_indices,minlength=len(notes_window_endpoints)-1)
        wins = [torch.arange(wl + weight.window_size, wh, 0.1) for wl, wh in zip(notes_window_endpoints[:-1], notes_window_endpoints[1:])]
        win_ics_notes = interpolator(wins)
        # # NOTE: it might be more appropriate to sum instead of mean, if the windows have different legnths, or better yet, do the max.
        ics_summed_notes = [win_ic.sum(axis=1).mean().item() for win_ic in win_ics_notes]
        # res.setdefault(piece_name, {})['n_notes'] = [ics_summed_notes, win_ics_notes, notes_window_counts]
        # pearsonr(ics_summed, notes_window_counts)
        # # score_positions = np.unique(onsets)
        # # ss = score_positions
        # ss = onsets
        # # NOTE: compute tonaltension
        # w_size = 1.0
        # wsl = np.ones(len(ss), dtype=np.float32)*w_size / 2
        # wsh = np.ones(len(ss), dtype=np.float32)*w_size / 2
        # ws = np.stack([wsl,wsh])
        
        # measure_changes,  = np.where(np.diff(matched_array['mn'], prepend=-1))
        ws = []
        ss = []
        for m in np.unique(matched_array['mn']):
             notes_in_measure = matched_array[matched_array['mn'] == m]
             min_time = min(notes_in_measure['onset_sec'])
             max_time = max(notes_in_measure['onset_sec'])
             mid = (min_time + max_time) / 2
             ws.append((mid-min_time, max_time-mid))
             ss.append(mid)
        ss = np.array(ss)
        wsl, wsh = zip(*ws)
        ws = np.stack(ws, axis=0)
        # wins = [(matched_array[i-1]['onset_sec'], matched_array[i]['onset_sec']) for i in range(1, len(measure_changes))]
        tt = pt.musicanalysis.estimate_tonaltension(matched_array, ss=ss, ws=ws)
        # NOTE: aggregate results bar level..
    

        # offset_comp_for_ic_memory = weight.window_size
        offset_comp_for_ic_memory = 0.0
        wins = [torch.arange(o-wl + offset_comp_for_ic_memory, o+wh, 0.1) for o, wl, wh in zip(ss, wsl, wsh)]
        win_ics_tt = interpolator(wins)
        # NOTE: it might be more appropriate to sum instead of mean, if the windows have different legnths, or better yet, do the max.
        ics_summed_tt = [win_ic.sum(axis=1).mean().item() for win_ic in win_ics_tt]
        # res.setdefault(piece_name, {})['tt'] = [ics_summed, win_ics, tt]
        # names = [*(tt.dtype.names[1:])]
        # data = [tt[n] for n in names]
        # names.append('ic')
        # data.append(np.array(ics_summed,np.float32))
        # df = pd.DataFrame(np.lib.recfunctions.append_fields(matched_array, names=names, data=data).filled())
        # return (ics_summed, *[tt[feat].tolist() for feat in ['cloud_diameter', 'cloud_momentum', 'tensile_strain']])
        return ics_summed_notes, win_ics_notes, notes_window_counts,  ics_summed_tt, win_ics_tt, tt