import dataclasses
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
from warnings import warn
from slugify import slugify
import torch
import os

import tqdm
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.dataset_managers.piano_midi_dataset import PianoMidiDataset
from CIA.getters import get_handler, get_data_processor, \
    get_dataloader_generator, get_decoder, get_positional_embedding,\
    get_sos_embedding
import time
import importlib
from CIA.ic import ICCurve, ICRes, TimepointsGenerator, Weight, unique_timepoints, DrawnICCurve, Piece
# from CIA.ic import get_mov_avg
from CIA.positional_embeddings.positional_embedding import PositionalEmbedding
from torch.nn.parallel import DistributedDataParallel
import einops
from CIA.utils import get_free_port
from CIA.data_processors.data_processor import DataProcessor
from CIA.handlers.decoder_events_handler import DecoderEventsHandler
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
from jsonargparse import ActionConfigFile, ArgumentParser, CLI, class_from_function
import logging
import multiprocessing
# import plotly.io as pio
# pio.renderers.default = "vscode"
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

model_dir = 'models/piano_event_performer_2021-10-01_16:03:06'
    # before_nodes: int
    # after_nodes: int

@dataclasses.dataclass
class Data(Iterable[Tuple[torch.LongTensor, str, int, Optional[Piece]]]):
    label: str
    @property
    def dataloader_generator(self) -> DataloaderGenerator:
        return self.dataloader_generator_
    @dataloader_generator.setter
    def dataloader_generator(self, val) -> None:
        self.dataloader_generator_ = val
    def __len__(self) -> int:
        raise NotImplementedError
    def __iter__(self):
        raise NotImplementedError


@dataclasses.dataclass
class DataCache(Data):
    # dataloader_generator : DataloaderGenerator
    n_inpaint : int
    n_pieces: Optional[int] = None
    def __post_init__(self):
        assert self.label in ['train', 'validation', 'test']
        assert self.label  != 'test',   'There\'s some problem with the test set' 
        assert self.n_inpaint < 512 - 5
    @Data.dataloader_generator.setter
    def dataloader_generator(self, val) -> None:
        # super().dataloader_generator.fset(self, val)
        # super(DataCache, self).dataloader_generator.fset(self, val)
        self.dataloader_generator_ = val
        ret = self.dataloader_generator.dataloaders(batch_size = 1, shuffle_val=True)
        val.dataset.split = self.label 
        dl_idx = {
            'train': 0,
            'validation': 1,
            'test': 2
        }[self.label]
        self.dataloader = ret[dl_idx]
    def __iter__(self):
        for i, original_x in enumerate(self.dataloader):
            if i == self.n_pieces:
                return
            piece_name = str(i)
            piece = None
            yield original_x['x'], piece_name, self.n_inpaint, piece
    def __len__(self) -> int:
        return len(self.dataloader_generator.dataset) if self.n_pieces is None else self.n_pieces
        
@dataclasses.dataclass
class DataPiece(Data):
    pieces : Iterable[Piece]
    def __len__(self) -> int:
        return len(self.pieces)
    def __repr__(self) -> str:
        return f'DataPiece({self.label})'
    def __iter__(self):
        for piece in self.pieces:
            piece_name = Path(piece.path).stem + f'_start_{piece.start_node}_nodes_{piece.n_inpaint}'
            # NOTE: parallelize over number of samples per piece
            ds : PianoMidiDataset = self.dataloader_generator.dataset
            sequence = ds.process_score(piece.path)
            orig_seq_length = len(sequence['pitch'])
            if piece.start_node > 0:
                sequence = {k : v[piece.start_node:] for k,v in sequence.items()}
            if piece.start_node < 0:
                raise NotImplementedError('Tetst that indeed it works')
            sequence = ds.add_start_end_symbols(
                sequence, start_time=piece.start_node, sequence_size=ds.sequence_size
            )
            # Tokenize
            sample = ds.tokenize(sequence)
            x = torch.tensor([sample[e] for e in self.dataloader_generator.features])
            original_x = einops.rearrange(x, 'f n -> 1 n f')
            yield original_x, piece_name, piece.n_inpaint, piece
        # raise StopIteration

@dataclasses.dataclass
class Experiment:
    time_points_generator: TimepointsGenerator
    weight : Weight
    dataset : Data
    ic_curve: Optional[ICCurve]
    # NOTE: here we should have either the test set, or some named collection of pieces....
@dataclasses.dataclass
class Config:
    k_traces : int
    samples_per_template: int
    logging: str
    experiment: Experiment
    def __post_init__(self):
        # TODO: exp uniquely identifies where
        args_str =  f'{slugify(str(self.experiment))}/k-traces-{str(self.k_traces)}'
        self.out = Path(f'out/{args_str}')
        self.out.mkdir(parents=True, exist_ok=True)
        numeric_level = getattr(logging, self.logging.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % self.logging)
        log_file = self.out.joinpath('log.txt')
        logging.basicConfig(filename=log_file, level=numeric_level)


def main(c :Config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        world_size = 1
        rank = 0
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = str(get_free_port())
    # if torch.cuda.is_available():
    #     torch.distributed.init_process_group(backend='nccl')
    # else:
    #     torch.distributed.init_process_group(backend='gloo')
    # rank = torch.distributed.get_rank()
    # world_size = torch.distributed.get_world_size()
    # if torch.distributed.is_initialized():
    #     rank = torch.distributed.get_rank()
    #     world_size = torch.distributed.get_world_size()
    # else:
    #     rank = 0
    #     world_size = 1
    # if torch.cuda.is_available() and c.GPU in list(range(4))
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        device_ids = [rank]
        output_device = rank
        device = f"cuda:{rank}"
    else:
        # cpu case
        torch.distributed.init_process_group(backend="gloo", world_size=world_size, rank=rank)
        device_ids = None
        output_device = None
        device = "cpu"
    # config =  importlib.import_module('CIA.configs.piarceiverStack').config
    # NOTE: override configuration
    config = importlib.import_module('.config_autoreg', f'{model_dir.replace("/", ".")}').config
    for t in ['time_dilation', 'velocity_shift', 'transposition']:
        config['dataloader_generator_kwargs']['transformations'][t] = False
    # NOTE: unconmment, since we indeed want to do only inpainting
    # config['data_processor_kwargs']['num_events_before'] = 1
    # config['data_processor_kwargs']['num_events_after'] = 0
    # TODO: check correct number...
    # num_events_middle = 1024 - 6
    # num_events_middle = 512 - 5 # magic - 5 for special tokens
    # num_events_middle = 256
    num_max_generated_events = 1024 - 6

    load = True
    overfitted = True

    # === Decoder ====
    # dataloader generator
    dataloader_generator = get_dataloader_generator(
        dataset=config["dataset"],
        dataloader_generator_kwargs=config["dataloader_generator_kwargs"],
    )

    # data processor
    data_processor: DataProcessor = get_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_type=config["data_processor_type"],
        data_processor_kwargs=config["data_processor_kwargs"],
    )

    # positional embedding
    positional_embedding: PositionalEmbedding = get_positional_embedding(
        dataloader_generator=dataloader_generator,
        positional_embedding_dict=config["positional_embedding_dict"],
    )

    # sos embedding
    sos_embedding = get_sos_embedding(
        dataloader_generator=dataloader_generator,
        sos_embedding_dict=config["sos_embedding_dict"],
    )

    decoder = get_decoder(
        data_processor=data_processor,
        dataloader_generator=dataloader_generator,
        positional_embedding=positional_embedding,
        sos_embedding=sos_embedding,
        decoder_kwargs=config["decoder_kwargs"],
        training_phase=False,
        handler_type=config["handler_type"],
    )

    decoder.to(device)
    decoder = DistributedDataParallel(
        module=decoder,
        device_ids=[rank] if device != 'cpu' else None,
        output_device=rank
        # )
        ,
        find_unused_parameters=True,
    )

    decoder_handler = get_handler(
        handler_type=config["handler_type"],
        decoder=decoder,
        model_dir=model_dir,
        dataloader_generator=dataloader_generator,
    )

    if load:
        if overfitted:
            decoder_handler.load(early_stopped=False)
        else:
            decoder_handler.load(early_stopped=True)

    if hasattr(decoder_handler.model.module.transformer, "fix_projection_matrices_"):
        decoder_handler.model.module.transformer.fix_projection_matrices_()
    # NOTE: parallelize over the pieces 
    # for piece in tqdm.tqdm(c.pieces, desc='Pieces completed', disable=rank != 0, position=0):
    # TODO: this is quite ugly, but how can it be done prettyer using jsonargparse?
    c.experiment.dataset.dataloader_generator = dataloader_generator
    for original_x, piece_name, n_inpaint, piece in tqdm.tqdm(c.experiment.dataset, desc='Pieces completed', disable=rank != 0, position=0):
        piece_folder = c.out.joinpath(piece_name)
        piece_folder.mkdir(exist_ok=True, parents=True)
        
        for i in tqdm.tqdm(np.arange(rank, c.samples_per_template, world_size), desc='sample pr piece', disable=rank != 0, position=1):
            
            file_folder = piece_folder.joinpath(f'{i}')
            file_folder.mkdir(exist_ok=True, parents=True)
            gen_file = file_folder.joinpath(f'ic.pt')
            if gen_file.exists():
                continue
            x, metadata_dict = data_processor.preprocess(original_x, num_events_middle=n_inpaint)
            placeholder_duration = metadata_dict['placeholder_duration'].item()
            ts = c.experiment.time_points_generator(x, metadata_dict)
            # TODO: ugly to check like this. Alternatively we could require that
            # the times are always relative. However, would be problematic for matching,
            # since the absolute time differences are important for the scaling.
            if c.experiment.ic_curve is not None:
                if isinstance(c.experiment.ic_curve, DrawnICCurve):
                    ic_curve : DrawnICCurve = c.experiment.ic_curve
                    ic_curve.set_placeholder_length(placeholder_duration)

            # # "open ended"
            # secs_dec = 25.
            # batch_size = 1
            # placeholder, placeholder_duration_token = data_processor.compute_placeholder(placeholder_duration=torch.tensor([secs_dec]), batch_size=batch_size)
            # before = einops.repeat(data_processor.start_tokens, 'd ->b 1 d', b=batch_size)
            # after = torch.empty(batch_size, 0, placeholder.shape[-1], dtype=torch.long, device=device)
            # sod =  einops.repeat(data_processor.start_tokens, 'd ->b 1 d', b=batch_size)
            # query = torch.cat([before, placeholder, after, sod], dim=1)
            # middle = torch.zeros(batch_size, num_events_middle-1, placeholder.shape[-1], dtype=torch.long, device=device)
            # end = einops.repeat(data_processor.end_tokens, 'd ->b 1 d', b=batch_size)
            # padding_len = config['dataloader_generator_kwargs']['sequences_size'] - query.shape[1] - middle.shape[1] - 1
            # padding = einops.repeat(data_processor.end_tokens, 'd ->b padding d', b=batch_size, padding=padding_len)
            # x = torch.cat([
            #     query,
            #     middle,
            #     end, 
            #     padding,

            # ],
            # axis=1)
            # metadata_dict = {
            #     'placeholder_duration': torch.tensor([secs_dec], device=device),
            #     'decoding_start': query.shape[1],
            #     # 'decoding_end': query.shape[1] - 1+  num_events_middle,
            # }


            # NOTE: with current setup this corresponds to open ended generation.
            # [START]  placeholder (programs the model to decode until a certain total time, where the end note is produced)  []  SOD  [m0,m1,...,m9]  END XX XX
            # NOTE: loss_mask is not used in inpainting, how do we handle attention mask?
            # metadata_dict.pop('loss_mask')
            # metadata_dict['decoding_start'] = 0
            # metadata_dict['decoding_end'] = 1024
            start_time = time.time()
            # NOTE: Here it always attends (!autoregressively!) to full sequence but updates during (autoregressive) decoding.
            # NOTE: model decodes from decode start and decode up to the next num_events_middle. 
            # If at one time the placeholder_duration is exceeded, or END symbol, then it terminates with done.
            # Otherwise it terminates after resampling all events in "middle".

            # NOTE: When generating all channels of a note, the prob is not autoreg, instead the state of previous note is computed, 
            # and the model is "programed" to generate a certain token in the head, by appending previous computed embeddings to he state
            # 
            #  If it did not . We can probably fix this 
            # by giving a sequence of some mask token, from the beginning. 
            # k_traces = 128
            # import torch.autograd.profiler as profiler
            # with profiler.profile(profile_memory=True, use_cuda=True, record_shapes=True) as prof:
            try:
                (
                    temp,
                    gen
                ) = decoder_handler.inpaint_ic_curve(
                    x=x.clone(),
                    interpolation_time_points=ts,
                    k_traces=c.k_traces,
                    weight_fn=c.experiment.weight,
                    metadata_dict=metadata_dict,
                    piece=piece,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    interpolator_template=c.experiment.ic_curve,
                    # num_max_generated_events=num_max_generated_events
                    num_max_generated_events=None
                )
                end_time = time.time()
                gen.write(gen_file)
                if i == 0 and rank == 0:
                    file_folder = piece_folder.joinpath('temp')
                    file_folder.mkdir(exist_ok=True, parents=True)
                    temp.write(file_folder.joinpath(f'ic.pt'))
            except Exception as e:
                logger = multiprocessing.get_logger()
                logger.error(f"Failed generating sample {piece_name} with expection {e}")

        # x_inpainted = torch.cat([before, generated_region, after, end], axis=1)
        # x_inpainted = data_processor.postprocess(x_gen, decoding_end, metadata_dict)
        # x_inpainted_orig = data_processor.postprocess(original_x, decoding_end, metadata_dict)
        # print(f"Time of generated sequence {dataloader_generator.get_elapsed_time(x_inpainted[0][None])[0,-1]}")
        # dataloader_generator.write(x_inpainted_orig[0], 'out/orig')
def express_to_suplot(fig_plotly, explot, row, col):
    for trace in explot.data:
        trace.showlegend = False
        fig_plotly.add_trace(trace, row=row, col=col)
def plot(c : Config):
    config = importlib.import_module('.config_autoreg', f'{model_dir.replace("/", ".")}').config
    for t in ['time_dilation', 'velocity_shift', 'transposition']:
        config['dataloader_generator_kwargs']['transformations'][t] = False
    dataloader_generator = get_dataloader_generator(
        dataset=config["dataset"],
        dataloader_generator_kwargs=config["dataloader_generator_kwargs"],
    )

    # data processor
    data_processor: DataProcessor = get_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_type=config["data_processor_type"],
        data_processor_kwargs=config["data_processor_kwargs"],
    )
    for temp_file in [f for f in c.out.rglob('*/temp/ic.pt')]:
        song_dir = temp_file.parent.parent
        res_temp = ICRes.load(p=temp_file)
        num_middle_tokens = res_temp.decoding_end-1 - (data_processor.num_events_after+data_processor.num_events_before+2)
        temp_midi = temp_file.parent.joinpath(f'song.mid')
        ds : PianoMidiDataset = dataloader_generator.dataset
        sequence = ds.process_score(res_temp.piece.path)
        sequence = ds.tokenize(sequence)
        sequence = torch.tensor([sequence[e] for e in dataloader_generator.features])
        sequence = einops.rearrange(sequence, 'f n -> n f')
        before = sequence[:data_processor.num_events_before+res_temp.piece.start_node]
        after = sequence[res_temp.piece.start_node+data_processor.num_events_before+num_middle_tokens:]
        dataloader_generator.write(sequence, temp_midi.parent.joinpath(temp_midi.stem))
        for sample in song_dir.rglob('*/ic.pt'):
            # try:
                if sample == temp_file:
                    continue
                res_gen = ICRes.load(p=sample)
                decoding_start = data_processor.num_events_after+data_processor.num_events_before+2
                middle_tokens_gen = res_gen.tok[decoding_start:res_gen.decoding_end-1]
                s = torch.cat([before, middle_tokens_gen, after], axis=0)
                gen_midi = sample.parent.joinpath(f'song.mid')
                dataloader_generator.write(s, gen_midi.parent.joinpath(gen_midi.stem))

                files = [temp_midi,gen_midi]
                res = [res_temp, res_gen]
                figs_pr_sample = 5
                titles = [title + f" {type_}" for type_ in ['template', 'generated'] for title in (
                    'Piano roll',
                    'Entr tokens',
                    'IC tokens',
                    'IC Interpolation',
                    'IC Interpolation summed channels'
                ) ] + ['IC Deviation']

                fig_plotly = make_subplots(
                    rows=len(files) * figs_pr_sample + 1,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=titles,
                    )
                # ic_tok_max = max([r.ic_tok.max() for r in res])
                ic_int_max = max([r.ic_int.max() for r in res])
                ic_int_summed_max = max([r.ic_int.sum(-1).max() for r in res])
                cum_ics_list = []
                entrs_list = []

                for i, (f, r) in enumerate(zip(files, res)):
                    midi = pretty_midi.PrettyMIDI(str(f))
                    # sr = 10
                    sr = 150 # 2 / .02, where 0.02 is the smallest time-shift
                    piano_roll = midi.get_piano_roll(sr).T
                    "/share/hel/home/mathias/.cache/mutdata/pia/databases/Piano/transcriptions/midi/Wagner, Richard, Ankunft bei den schwarzen Schwänen, WWV 95, 83pIdDPBQg4.mid"
                    end = piano_roll.shape[0]/sr

                    pitch_range = (24,96)
                    image = px.imshow(
                        # TODO: find better way to do this which keeps the correct values shown...
                        np.log10(piano_roll.T + 1e-6),
                        origin="lower",
                        color_continuous_scale="plasma",
                        labels=dict(x="Time", y="Pitch", color="Velocity"),
                        x=np.arange(0, end, 1/sr),
                        )
                    # TODO: this low level time calculation of tokens should not be here.
                    time_before = torch.sum(torch.tensor(
                        [dataloader_generator.dataset.index2value['time_shift'][tok[3].item()] for tok in before]
                    ), dim=0).item()
                    channels = slice(0, 4)
                    n_channels = channels.stop- channels.start
                    middle_tokens = r.tok[decoding_start:r.decoding_end-1]
                    shifts_middle = [dataloader_generator.dataset.index2value['time_shift'][tok[3].item()] for tok in middle_tokens]
                    time_middle = torch.sum(torch.tensor(
                        shifts_middle
                    ), dim=0).item() + time_before
                    express_to_suplot(fig_plotly, image, row=i *figs_pr_sample + 1, col=1)
                    fig_plotly.add_shape(
                        go.layout.Shape(
                            type="line",
                            yref="paper",
                            xref="x",
                            x0=time_before,
                            y0=0,
                            x1=time_before,
                            y1=128,),
                        row=i *figs_pr_sample + 1,
                        col=1
                    )
                    fig_plotly.add_shape(
                        go.layout.Shape(
                            type="line",
                            yref="paper",
                            xref="x",
                            x0=time_middle,
                            y0=0,
                            x1=time_middle,
                            y1=128,),
                        row=i *figs_pr_sample + 1,
                        col=1
                    )
                    fig_plotly.update_yaxes(range=pitch_range, row=i *figs_pr_sample + 1, col=1)  
                    n_channels = channels.stop- channels.start
                    if r.ic_tok is not None:
                        n_points = len(r.timepoints)
                        c_ = np.broadcast_to(np.array(dataloader_generator.features)[None,:], (n_points, n_channels)).flatten()
                        times = np.broadcast_to(np.array(r.timepoints + time_before )[:,None], (n_points, n_channels)).flatten()
                        # times = r.timepoints + time_before 
                        entrs_list.append(r.entr_tok.max())                 
                        scatter = px.scatter(
                            x=times,
                            y=r.entr_tok[:, channels].numpy().flatten(),
                            color=c_,
                            # color_discrete_sequence=['red', 'green', 'blue'],
                            labels=dict(x="Time", y="Entropy", color="Channel"),
                            # color_continuous_scale="plasma",
                            )
                        express_to_suplot(fig_plotly, scatter, row=i *figs_pr_sample + 2, col=1)                        # pass
                        # unique_timepoints_, cum_ics = unique_timepoints(r.timepoints, r.ic_tok)
                        # unique_timepoints_, cum_ics = r.timepoints, r.ic_tok
                        cum_ics_list.append(r.ic_tok.max())
                        # unique_timepoints_ += time_before                     
                        scatter = px.scatter(
                            x=times,
                            y=r.ic_tok[:, channels].numpy().flatten(),
                            color=c_,
                            # color_discrete_sequence=['red', 'green', 'blue'],
                            labels=dict(x="Time", y="IC", color="Channel"),
                            # color_continuous_scale="plasma",
                            )
                        express_to_suplot(fig_plotly, scatter, row=i *figs_pr_sample + 3, col=1)
                    ts = r.timepoints_int + time_before
                    ts_b = np.broadcast_to(ts[:,None], r.ic_int.shape).flatten()
                    colors = np.broadcast_to(np.array(dataloader_generator.features)[None, :], r.ic_int.shape).flatten()
                    express_to_suplot(fig_plotly, px.line(x=ts_b, y=r.ic_int.flatten(), color=colors, line_shape='hv', labels=dict(x="Time", y="IC", color="Channel")), row=i *figs_pr_sample + 4, col=1)
                    int_summed_channels = r.ic_int.sum(-1)
                    express_to_suplot(fig_plotly, px.line(x=ts, y=int_summed_channels, line_shape='hv', labels=dict(x="Time", y="IC", color="Channel")), row=i *figs_pr_sample + 5, col=1)
                ic_dev = res_gen.ic_dev
                ic_dev_sum = ic_dev.sum().item()
                express_to_suplot(fig_plotly, px.line(x=ts, y=ic_dev, line_shape='hv', labels=dict(x="Time", y=f"IC Deviation", color="Channel")), row=i *figs_pr_sample + 6, col=1)
                fig_plotly.layout.annotations[10]['text'] = f"IC Deviation , sum={ic_dev_sum}"
                cum_ics_max = max(cum_ics_list)
                entrs_max = max(entrs_list)
                for i in range(2):
                    fig_plotly.update_yaxes(range=(0, entrs_max+1), row=i *figs_pr_sample + 2, col=1) 
                    fig_plotly.update_yaxes(range=(0, cum_ics_max+1), row=i *figs_pr_sample + 3, col=1) 
                    fig_plotly.update_yaxes(range=(0, ic_int_max+1), row=i *figs_pr_sample + 4, col=1)
                    fig_plotly.update_yaxes(range=(0, ic_int_summed_max+1), row=i *figs_pr_sample + 5, col=1)
                fig_height = 2000
                fig_plotly.update_layout(height=fig_height)
                fig_plotly.write_html(sample.parent.joinpath('ic_curve.html'))
            # except Exception as e:
            #     logger = multiprocessing.get_logger()
            #     logger.error(f"Failed with expection {e}")

def eval_(configs : List[Config]):
    import pandas as pd
    # import wandb
    # warn('Update config')
    # wandb.init(
    #     project="ic_gen",
    #     config={'k_traces': c.k_traces},
    #     group=str(c.experiment)
    # )
    # wandb.run.summary['test'] = 1
    result = []
    exps = []
    params = []
    pieces = []
    samples = []
    times = []
    ic_devs = []
    for c in configs:
        for piece_dir in c.out.glob('*'):
            if piece_dir.is_dir():            
                for sample in piece_dir.rglob('*/ic.pt'):
                    if sample.parent.name != 'temp':
                        res = ICRes.load(p=sample)
                        exps.extend(len(res.ic_dev)*[str(c.experiment)])
                        # TODO: make more general
                        params.extend(len(res.ic_dev)*[c.k_traces])
                        pieces.extend(len(res.ic_dev)*[piece_dir.name])
                        samples.extend(len(res.ic_dev)*[sample.parent.name])
                        times.extend(res.timepoints_int.numpy())
                        ic_devs.extend(res.ic_dev.numpy())
        result= pd.DataFrame({
            'exps': pd.Categorical(exps),
            'params': pd.Series(params),
            'piece' : pd.Series(pieces),
            'sample' : pd.Series(samples),
            'time' : pd.Series(times),
            'ic_dev' : pd.Series(ic_devs)
        })
    # grouped = result.groupby(['exps', 'params', 'piece']).apply(lambda x: x.pivot(columns='time', values='ic_dev', index='sample'))
    # grouped.loc[str(result.exps.iloc[0])].mean(1)
    # df = result.groupby('params').mean()
    df = result.groupby(['exps', 'params']).agg({'ic_dev': ['mean', 'std', 'count']})
    print(df)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--app", type=Config, nargs='*')  
    parser.add_argument("--app", type=List[Config])  
    parser.add_argument("--config", action=ActionConfigFile)
    # NOTE: needed for torch.distributed.launch
    parser.add_argument("--local_rank", type=int, default=None)
    subcommands = parser.add_subcommands()
    main_subcomm = ArgumentParser()
    subcommands.add_subcommand("main", main_subcomm)
    plot_subcomm = ArgumentParser()
    subcommands.add_subcommand("plot", plot_subcomm)
    eval_subcomm = ArgumentParser()
    subcommands.add_subcommand("eval", eval_subcomm)
    args = parser.parse_args()
    init = parser.instantiate_classes(args)
    if args.subcommand in ['main', 'plot']:
        if len(args.config) > 1:
                raise NotImplementedError
        else:
            c : Config = init.app[0]
        if args.subcommand == "main":
            if len(args.config) > 1:
                raise NotImplementedError
            else:
                c : Config = init.app[0]
            if 'RANK' not in  os.environ or int(os.environ['RANK']) == 0 :
                dir = Path(c.out)
                dir.mkdir(exist_ok=True, parents=True)
                parser.save(args, dir.joinpath('config.yaml'), overwrite=True)
            main(c)
            # NOTE: allow the processes to finish before plotting
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if 'RANK' not in os.environ or int(os.environ['RANK']) == 0 :    
                plot(c)

        elif args.subcommand == "plot":
            plot(c)
    elif args.subcommand == "eval":
        eval_(init.app)
    else:
        raise ValueError(f"Unknown subcommand {args.subcommand}")

