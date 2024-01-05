import copy
from pathlib import Path
from typing import List
from warnings import warn
import torch
import os

import tqdm
from CIA.dataset_managers.piano_midi_dataset import PAD_SYMBOL, PianoMidiDataset
from CIA.getters import get_handler, get_data_processor, \
    get_dataloader_generator, get_decoder, get_positional_embedding,\
    get_sos_embedding
import time
import importlib
from CIA.ic import DataPiece, ICRes, DrawnICCurve
from CIA.positional_embeddings.positional_embedding import PositionalEmbedding
from torch.nn.parallel import DistributedDataParallel
import einops
from CIA.utils import get_free_port
from CIA.data_processors.data_processor import DataProcessor
import numpy as np
import pretty_midi
from jsonargparse import ActionConfigFile, ArgumentParser
import multiprocessing
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from ic.app import Config

from ic.eval_ import eval_

model_dir = 'models/piano_event_performer_2021-10-01_16:03:06'

        # logging.getLogger()


def main(c : Config, device='cpu'):
    logger = multiprocessing.get_logger()
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
    for original_x, piece_name, n_inpaint, piece in tqdm.tqdm(c.experiment.dataset, desc='Pieces completed', disable=rank != 0, position=0, leave=True):
        piece_folder = c.out.joinpath(piece_name)
        piece_folder.mkdir(exist_ok=True, parents=True)
        for i in tqdm.tqdm(np.arange(rank, c.samples_per_template, world_size), desc='sample pr piece', disable=rank != 0, position=1, leave=False):
            file_folder = piece_folder.joinpath(f'{i}')
            file_folder.mkdir(exist_ok=True, parents=True)
            gen_file = file_folder.joinpath(f'ic.pt')
            if gen_file.exists():
                continue
            if isinstance(n_inpaint, float):
                rest = original_x[
                    :, data_processor.num_events_before : 
                ]
                # TODO: rewrite for clearness, removes padding
                rest_ = rest[:, rest[0,:,3] != 105, :]
                if rest_.numel() == 0:
                    logger.error(f"Skipping {piece_name} since its all padding")
                    continue
                durations = dataloader_generator.get_elapsed_time(rest_)
                n_inpaint_ = (durations[0]-n_inpaint).abs().argmin().item()
                if n_inpaint_ <= 1 or (durations == 0).all():
                    logger.error(f"Skipping {piece_name} since it has no duration")
                    continue
            elif isinstance(n_inpaint, (int, np.integer)):
                n_inpaint_ = n_inpaint
            x, metadata_dict = data_processor.preprocess(original_x, num_events_middle=n_inpaint_)
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
            # try:
            (
                temp,
                gen
            ) = decoder_handler.inpaint_ic_curve(
                x=x.clone(),
                interpolation_time_points=ts,
                sampling_config=c.sampling_config,
                experiment=c.experiment,
                metadata_dict=metadata_dict,
                piece=piece,
                top_p=1.0,
                top_k=0,
                
                # num_max_generated_events=num_max_generated_events
                num_max_generated_events=None
            )
            gen.write(gen_file)
            if i == 0 and rank == 0:
                file_folder = piece_folder.joinpath('temp')
                file_folder.mkdir(exist_ok=True, parents=True)
                temp.write(file_folder.joinpath(f'ic.pt'))
            # except Exception as e:
            #     logger.error(f"Failed generating sample {piece_name} with expection {e}")

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
    # TODO: change all naming to metric
    if not isinstance(c.experiment.dataset, DataPiece):
        warn('Check implimentation, since it is not clear how to handle this case')
        # raise NotImplementedError
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
        if temp_midi.exists():
            continue
        ds : PianoMidiDataset = dataloader_generator.dataset
        decoding_start = data_processor.num_events_after+data_processor.num_events_before+2
        if res_temp.piece is not None:
            sequence = ds.process_score(res_temp.piece.path)
            sequence = ds.tokenize(sequence)
            sequence = torch.tensor([sequence[e] for e in dataloader_generator.features])
            sequence = einops.rearrange(sequence, 'f n -> n f')
            before = sequence[:data_processor.num_events_before+res_temp.piece.start_node]
            after = sequence[res_temp.piece.start_node+data_processor.num_events_before+num_middle_tokens:]
            dataloader_generator.write(sequence, temp_midi.parent.joinpath(temp_midi.stem))
        else:
            before = res_temp.tok[:data_processor.num_events_before]
            after = res_temp.tok[data_processor.num_events_before+1: decoding_start-1]
            middle_tokens_temp = res_temp.tok[decoding_start:res_temp.decoding_end-1]
            sequence = torch.cat([before, middle_tokens_temp, after], axis=0)
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
                metric = c.experiment.match_metric.capitalize()
                titles = [title + f" {type_}" for type_ in ['template', 'generated'] for title in (
                    'Piano roll',
                    'Entr tokens',
                    f'{metric} tokens',
                    f'{metric} Interpolation',
                    f'{metric} Interpolation summed channels'
                ) ] + [f'{metric} Deviation']

                fig_plotly = make_subplots(
                    rows=len(files) * figs_pr_sample + 1,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=titles,
                    )
                # ic_tok_max = max([r.ic_tok.max() for r in res])
                metric_int_max = max([r.ic_int.max() for r in res])
                metric_int_summed_max = max([r.ic_int.sum(-1).max() for r in res])
                cum_metric_list = []
                entrs_list = []

                for i, (f, r) in enumerate(zip(files, res)):
                    midi = pretty_midi.PrettyMIDI(str(f))
                    sr = 25
                    # sr = 150 # 2 / .02, where 0.02 is the smallest time-shift
                    piano_roll = midi.get_piano_roll(sr).T
                    end = piano_roll.shape[0]/sr

                    pitch_range = (24,96)
                    image = px.imshow(
                        # TODO: find better way to do this which keeps the correct values shown...
                        # np.log10(piano_roll.T + 1e-6),
                        piano_roll.T,
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
                        # NOTE: old functinality
                        if r.timepoints.dim() == 2:
                            times = np.array(r.timepoints + time_before ).flatten()
                        else:
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
                        cum_metric_list.append(r.ic_tok.max())
                        # unique_timepoints_ += time_before                     
                        scatter = px.scatter(
                            x=times,
                            y=r.ic_tok[:, channels].numpy().flatten(),
                            color=c_,
                            # color_discrete_sequence=['red', 'green', 'blue'],
                            labels=dict(x="Time", y=metric, color="Channel"),
                            # color_continuous_scale="plasma",
                            )
                        express_to_suplot(fig_plotly, scatter, row=i *figs_pr_sample + 3, col=1)
                    ts = r.timepoints_int + time_before
                    ts_b = np.broadcast_to(ts[:,None], r.ic_int.shape).flatten()
                    colors = np.broadcast_to(np.array(dataloader_generator.features)[None, :], r.ic_int.shape).flatten()
                    express_to_suplot(fig_plotly, px.line(x=ts_b, y=r.ic_int.flatten(), color=colors, line_shape='hv', labels=dict(x="Time", y=metric, color="Channel")), row=i *figs_pr_sample + 4, col=1)
                    int_summed_channels = r.ic_int.sum(-1)
                    express_to_suplot(fig_plotly, px.line(x=ts, y=int_summed_channels, line_shape='hv', labels=dict(x="Time", y=metric, color="Channel")), row=i *figs_pr_sample + 5, col=1)
                metric_dev = res_gen.ic_dev
                metric_dev_mean = metric_dev.mean().item()
                express_to_suplot(fig_plotly, px.line(x=ts, y=metric_dev, line_shape='hv', labels=dict(x="Time", y=f"{metric} Deviation", color="Channel")), row=i *figs_pr_sample + 6, col=1)
                fig_plotly.layout.annotations[10]['text'] = f"{metric} Deviation, mean={metric_dev_mean}"
                cum_metric_max = max(cum_metric_list)
                entrs_max = max(entrs_list)
                for i in range(2):
                    fig_plotly.update_yaxes(range=(0, entrs_max+1), row=i *figs_pr_sample + 2, col=1) 
                    fig_plotly.update_yaxes(range=(0, cum_metric_max+1), row=i *figs_pr_sample + 3, col=1) 
                    fig_plotly.update_yaxes(range=(0, metric_int_max+1), row=i *figs_pr_sample + 4, col=1)
                    fig_plotly.update_yaxes(range=(0, metric_int_summed_max+1), row=i *figs_pr_sample + 5, col=1)
                fig_height = 2000
                fig_plotly.update_layout(height=fig_height)
                fig_plotly.write_html(sample.parent.joinpath(f'{metric}_curve.html'))
                # fig_plotly.write_image(sample.parent.joinpath('ic_curve.svg'))
            # except Exception as e:
            #     logger = multiprocessing.get_logger()
            #     logger.error(f"Failed with expection {e}")

if __name__ == "__main__":
    parser = ArgumentParser(
        # default_config_files=['configs/config.yaml']
    )
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
    eval_subcomm.add_argument("--out_file", type=str, default='out/results/result.h5')
    subcommands.add_subcommand("eval", eval_subcomm)
    args = parser.parse_args()
    init = parser.instantiate_classes(args)
    if args.subcommand == 'main':
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(get_free_port())
            world_size = 1
            rank = 0
        if torch.cuda.is_available():
            torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
        else:
            # cpu case
            torch.distributed.init_process_group(backend="gloo", world_size=world_size, rank=rank)
            device = "cpu"
        for i, c in enumerate(init.app):
            if 'RANK' not in  os.environ or int(os.environ['RANK']) == 0 :
                dir = Path(c.out)
                dir.mkdir(exist_ok=True, parents=True)
                args_exp = copy.copy(args)
                args_exp.app = [args.app[i]]
                parser.save(args_exp, dir.joinpath('config.yaml'), overwrite=True)
            print(f'Experiment: {c.experiment}\n Sampling Config: {c.sampling_config}')
            main(c, device=device)
            # NOTE: allow the processes to finish before plotting
            # if torch.distributed.is_initialized():
            #     torch.distributed.barrier()
            # if 'RANK' not in os.environ or int(os.environ['RANK']) == 0 :    
            #     plot(c)

    elif args.subcommand == "plot":
        for c in init.app:
            plot(c)
    elif args.subcommand == "eval":
        eval_(init.app, init.eval.out_file)
    else:
        raise ValueError(f"Unknown subcommand {args.subcommand}")

