from dataclasses import asdict
import json
from ic.beam_search.io import ICRes
from ic.config import Config
#  present_df


import numpy as np
import torch
import tqdm


from typing import List
from warnings import warn

# def windowing():


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