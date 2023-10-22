import dataclasses
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple
from warnings import warn
from slugify import slugify
import torch
import os

import tqdm
from CIA.dataset_managers.piano_midi_dataset import PianoMidiDataset
from CIA.getters import get_handler, get_data_processor, \
    get_dataloader_generator, get_decoder, get_positional_embedding,\
    get_sos_embedding
import time
import importlib
from CIA.ic import ICRes, unique_timepoints
from CIA.ic import Piece
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

model_dir = 'models/piano_event_performer_2021-10-01_16:03:06'
    # before_nodes: int
    # after_nodes: int
@dataclasses.dataclass
class Config:
    pieces : Iterable[Piece]
    # GPU: int
    step : float
    weight : Callable[[torch.FloatTensor], torch.FloatTensor]
    k_traces : int
    samples_per_template: int
    # local_rank: Optional[int] = None
    def __post_init__(self):
        # TODO: this does not work in geneal, but to keep it simple for now
        args = dict(**self.weight.__dict__, step=self.step, k_traces=self.k_traces)
        args_str = slugify(str(tuple(sorted(args.items()))))
        self.out = Path(f'out/{args_str}')
        # NOTE: interpret as range
        # if isinstance(self.step, Tuple) and len(self.step) == 3:
        #     self.step = torch.arange(*self.step)
        # else:
        #     raise ValueError(f"ts must be tuple, got {type(self.step)}")


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
    num_events_middle = 256
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
    # pieces = [c.pieces[i] for i in range(rank, len(c.pieces), world_size)]
    for piece in tqdm.tqdm(c.pieces, desc='Pieces completed', disable=rank != 0, position=0):
        piece_name = Path(piece.path).stem
        piece_folder = c.out.joinpath(piece_name)
        piece_folder.mkdir(exist_ok=True, parents=True)
        # NOTE: parallelize over number of samples per piece
        ds : PianoMidiDataset = dataloader_generator.dataset
        sequence = ds.process_score(piece.path)
        sequence = ds.add_start_end_symbols(
            sequence, start_time=piece.start_node, sequence_size=ds.sequence_size
        )
        # Tokenize
        sample = ds.tokenize(sequence)
        x = torch.tensor([sample[e] for e in dataloader_generator.features])
        original_x = einops.rearrange(x, 'f n -> 1 n f')
        x, metadata_dict = data_processor.preprocess(original_x, num_events_middle=num_events_middle)
        for i in tqdm.tqdm(np.arange(rank, c.samples_per_template, world_size), desc='sample pr piece', disable=rank != 0, position=1):
            # dataloader_generator.dataset.split = split
            # data = dataloader_generator.dataset[id_]
            # original_x = torch.stack([data[e] for e in dataloader_generator.features], dim=-1)[None]
            # NOTE: num_events_middle likely the number of notes which will be created? Actually more events are created...
            # NOTE: first event is always the same, probably because it's a special start token, .
            # NOTE: Num events, most likely, defines the event (after decode start) that should be how long time (in the original score) we should we should generate. 
            # TODO: what's the difference between num_events_middle and num_max_generated_events? Probably 
            # Only needed for setting up start_decode and end_decode.
            # Does the model actually actually not attend to the after (future) tokens while decoding? 
            # Actually preprocess should return (shuffle the data): 
            # before  middle  after -> before  placeholder  after  SOD (start of decoding)  middle  END XX XX (pad)
            # NOTE: metadata_dict['decoding_end'] is not used for inpaint_non_optimized 
            # dataloader_generator.dataset.process_score('/share/home/mathias/.cache/mutdata/pia/databases/Piano/transcriptions/midi/Bach, Carl Philipp Emanuel, Keyboard Sonata in F 
            x, metadata_dict = data_processor.preprocess(original_x, num_events_middle=num_events_middle)
            ts = torch.arange(0, metadata_dict['placeholder_duration'].item(), c.step)
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
            
            (
                temp,
                gen
            ) = decoder_handler.inpaint_ic_curve(
                x=x.clone(),
                interpolation_time_points=ts,
                k_traces=c.k_traces,
                weight_fn=c.weight,
                metadata_dict=metadata_dict,
                piece=piece,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                num_max_generated_events=num_max_generated_events
            )
            end_time = time.time()
            file_folder = piece_folder.joinpath(f'{i}')
            file_folder.mkdir(exist_ok=True, parents=True)
            gen.write(file_folder.joinpath(f'ic.pt'))
        file_folder = piece_folder.joinpath('temp')
        file_folder.mkdir(exist_ok=True, parents=True)
        temp.write(file_folder.joinpath(f'ic.pt'))

        # x_inpainted = torch.cat([before, generated_region, after, end], axis=1)
        # x_inpainted = data_processor.postprocess(x_gen, decoding_end, metadata_dict)
        # x_inpainted_orig = data_processor.postprocess(original_x, decoding_end, metadata_dict)
        # print(f"Time of generated sequence {dataloader_generator.get_elapsed_time(x_inpainted[0][None])[0,-1]}")
        # dataloader_generator.write(x_inpainted_orig[0], 'out/orig')
def plot(c : Config):
    config = importlib.import_module('.config_autoreg', f'{model_dir.replace("/", ".")}').config
    for t in ['time_dilation', 'velocity_shift', 'transposition']:
        config['dataloader_generator_kwargs']['transformations'][t] = False
    # config['data_processor_kwargs']['num_events_before'] = 1
    # config['data_processor_kwargs']['num_events_after'] = 0
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
    for song_dir in [f for f in c.out.glob('*/') if f.is_dir()]:
        temp_file = song_dir.joinpath('temp/ic.pt')
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
            if sample == temp_file:
                continue
            res_gen = ICRes.load(p=sample)
            decoding_start = data_processor.num_events_after+data_processor.num_events_before+2
            middle_tokens = res_gen.tok[decoding_start:res_gen.decoding_end-1]
            s = torch.cat([before, middle_tokens, after], axis=0)
            gen_midi = sample.parent.joinpath(f'song.mid')
            dataloader_generator.write(s, gen_midi.parent.joinpath(gen_midi.stem))

            files = [temp_midi,gen_midi]
            res = [res_temp, res_gen]
            figs_pr_sample = 4
            fig, ax = plt.subplots(len(files) * figs_pr_sample, 1, figsize=(15,9), sharex=True)
            for i, (f, r) in enumerate(zip(files, res)):
                midi = pretty_midi.PrettyMIDI(str(f))
                # first_onset = min([n.start for n in  midi2.instruments[0].notes])
                # for note in midi2.instruments[0].notes:
                #     note.start -= first_onset
                #     note.end -= first_onset
                # delta = 1e-6
                # for cc in midi2.instruments[0].control_changes:
                #     cc.time -= first_onset
                #     if cc.time < 0:
                #         cc.time = 0
                sr = 10
                pr2 = midi.get_piano_roll(sr).T
                end = pr2.shape[0]/sr


                legends = dataloader_generator.features
                bs = 2

                nr = (24,96)
                # nr = (None, None)
                ax[i *figs_pr_sample + 0].imshow(
                    pr2.T[slice(*nr)],
                    origin='lower',
                    extent=[0, end, *nr],
                    aspect='auto',
                    interpolation='none',
                    cmap='plasma',
                )

                channels = slice(0, 4)
                n_channels = channels.stop- channels.start
                unique_timepoints_, cum_ics = unique_timepoints(r.timepoints, r.ic_tok)
                time_before = torch.sum(torch.tensor(
                    [dataloader_generator.dataset.index2value['time_shift'][tok[3].item()] for tok in before]
                ), dim=0).item()
                unique_timepoints_ += time_before
                n_points = len(unique_timepoints_)
                # ax[1].plot(unique_timepoint, cum_ics[:, channel],'.-')
                c_ = np.broadcast_to(np.arange(n_channels)[None,:], (n_points, n_channels)).flatten()

                times = np.broadcast_to(np.array(unique_timepoints_)[:,None], (n_points, n_channels)).flatten()
                scatter = ax[i*figs_pr_sample+1].scatter(
                    times,
                    cum_ics[:, channels].numpy().flatten(),
                    c=c_,
                    s=1.5,
                    marker='.'
                )
                legend1 = ax[i*figs_pr_sample+1].legend(*scatter.legend_elements(),
                                    loc="lower left", title="Classes")
                ax[i*figs_pr_sample+1].add_artist(legend1)

                warn('reimpliment these plots!')
                # ax[i*figs_pr_sample+2].step(c.tim, r.ic_int[:, channels], where='post')
                # ax[i*figs_pr_sample+2].legend(legends[channels])

                # ax[i*figs_pr_sample+3].step(c.step, r.ic_int[:, channels].sum(1), where='post')
                
                fig.tight_layout()
            plt.savefig(sample.parent.joinpath('ic_curve.pdf'))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--app", type=Config)  
    parser.add_argument("--config", action=ActionConfigFile)
    # NOTE: needed for torch.distributed.launch
    parser.add_argument("--local_rank", type=int, default=None)
    subcommands = parser.add_subcommands()
    parser_subcomm1 = ArgumentParser()
    subcommands.add_subcommand("main", parser_subcomm1)
    parser_subcomm2 = ArgumentParser()
    subcommands.add_subcommand("plot", parser_subcomm2)
    args = parser.parse_args()
    init = parser.instantiate_classes(args)
    c : Config = init.app
    if args.subcommand == "main":
        if 'RANK' not in  os.environ or int(os.environ['RANK']) == 0 :
            dir = Path(c.out)
            dir.mkdir(exist_ok=True, parents=True)
            parser.save(args, dir.joinpath('config.yaml'), overwrite=True)
        main(c)

    elif args.subcommand == "plot":
        plot(c)
    else:
        raise ValueError(f"Unknown subcommand {args.subcommand}")
    # c : Config = CLI(Config, as_positional=False)

