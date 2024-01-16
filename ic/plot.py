from CIA.data_processors.data_processor import DataProcessor
from CIA.dataloaders.piano_dataloader import PianoDataloaderGenerator
from CIA.dataset_managers.piano_midi_dataset import PianoMidiDataset
from CIA.ic import DataPiece, ICRes
from ic.app import Config

import einops
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pretty_midi
import torch
from plotly.subplots import make_subplots


from warnings import warn


def express_to_suplot(fig_plotly, explot, row, col):
    for trace in explot.data:
        trace.showlegend = False
        fig_plotly.add_trace(trace, row=row, col=col)


def plot(c : Config,
        #  dataloader_generator : PianoDataloaderGenerator,
          data_processor : DataProcessor
        ):
    # TODO: change all naming to metric
    if not isinstance(c.experiment.dataset, DataPiece):
        warn('Check implimentation, since it is not clear how to handle this case')
        # raise NotImplementedError
    # config = importlib.import_module('.config_autoreg', f'{model_dir.replace("/", ".")}').config
    # for t in ['time_dilation', 'velocity_shift', 'transposition']:
    #     config['dataloader_generator_kwargs']['transformations'][t] = False
    # dataloader_generator = get_dataloader_generator(
    #     dataset=config["dataset"],
    #     dataloader_generator_kwargs=config["dataloader_generator_kwargs"],
    # )

    # # data processor
    # data_processor: DataProcessor = get_data_processor(
    #     dataloader_generator=dataloader_generator,
    #     data_processor_type=config["data_processor_type"],
    #     data_processor_kwargs=config["data_processor_kwargs"],
    # )
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
            if sample == temp_file:
                continue
            res_gen = ICRes.load(p=sample)
            decoding_start = data_processor.num_events_after+data_processor.num_events_before+2
            middle_tokens_gen = res_gen.tok[decoding_start:res_gen.decoding_end-1]
            s = torch.cat([before, middle_tokens_gen, after], axis=0)
            gen_midi = sample.parent.joinpath(f'song.mid')
            dataloader_generator.write(s, gen_midi.parent.joinpath(gen_midi.stem))

            midi_files = [temp_midi, gen_midi]
            results = [res_temp, res_gen]
            metric_name = c.experiment.match_metric
            metric_name = metric_name.capitalize()
            figs_pr_sample, fig_plotly = get_figure(metric_name)
            
            
            for i, (midi_file, result) in enumerate(zip(midi_files, results)):
                midi = pretty_midi.PrettyMIDI(str(midi_file))
                # NOTE: calculate timepoints
                shift_to_time = dataloader_generator.dataset.index2value['time_shift']
                inpaint_time, times_tok, times_int_summed, times_int = calculate_times(shift_to_time, decoding_start, before, result)
                
                # NOTE: colors
                n_points = len(result.timepoints)
                features = np.array(dataloader_generator.features)
                colors_tok, colors_int = get_colors(features, n_points, result)
                
                # NOTE: figure rows
                img_offset = i*figs_pr_sample
                piano_roll_fig_row = img_offset+1
                entr_fig_row = img_offset + 2
                metric_fig_row = img_offset + 3
                int_fig_row = img_offset + 4
                int_summed_fig_row = img_offset + 5

                metric_dev_fig_row = 1 *figs_pr_sample + 6
            
                # NOTE: plot piano roll
                sr = 25
                # sr = 150 # 2 / .02, where 0.02 is the smallest time-shift
                piano_roll = midi.get_piano_roll(sr).T
                plot_piano_roll(fig_plotly, piano_roll_fig_row, sr, piano_roll, inpaint_time)
                # plot image after
                if result.ic_tok is not None:
                    # entrs_list.append(result.entr_tok.max())
                    entr = result.entr_tok.numpy().flatten()
                    plot_entr(fig_plotly, colors_tok, times_tok, entr, entr_fig_row)
                    
                    # cum_metric_list.append(result.ic_tok.max())    
                    metric = result.ic_tok.numpy().flatten()
                    plot_metric(metric, metric_name, fig_plotly, colors_tok, times_tok, metric_fig_row)
                express_to_suplot(fig_plotly, px.line(x=times_int, y=result.ic_int.flatten(), color=colors_int, line_shape='hv', labels=dict(x="Time", y=metric_name, color="Channel")), row=int_fig_row, col=1)
                int_summed_channels = result.ic_int.sum(-1)
                express_to_suplot(fig_plotly, px.line(x=times_int_summed, y=int_summed_channels, line_shape='hv', labels=dict(x="Time", y=metric_name, color="Channel")), row=int_summed_fig_row, col=1)
            plot_metric_dev(res_gen, metric_name, fig_plotly, times_int_summed, metric_dev_fig_row)
            
            # NOTE update y-axis
            # NOTE: 
            metric_int_max = max([r.ic_int.max() for r in results])
            metric_int_summed_max = max([r.ic_int.sum(-1).max() for r in results])
            # cum_metric_list = []
            # entrs_list = []
            entrs_max = max([r.entr_tok.max() for r in results])
            
            # cum_metric_max = max(cum_metric_list)
            cum_metric_max = max([r.ic_tok.max() for r in results])
            # entrs_max = max(entrs_list)
            for i in range(2):
                fig_plotly.update_yaxes(range=(0, entrs_max+1), row=entr_fig_row, col=1)
                fig_plotly.update_yaxes(range=(0, cum_metric_max+1), row=metric_fig_row, col=1)
                fig_plotly.update_yaxes(range=(0, metric_int_max+1), row=int_fig_row, col=1)
                fig_plotly.update_yaxes(range=(0, metric_int_summed_max+1), row=int_summed_fig_row, col=1)
            fig_height = 2000
            fig_plotly.update_layout(height=fig_height)
            fig_plotly.write_html(sample.parent.joinpath(f'{metric_name}_curve.html'))

def get_figure(metric):
    figs_pr_sample = 5
    titles = [title + f" {type_}" for type_ in ['template', 'generated'] for title in (
                'Piano roll',
                'Entr tokens',
                f'{metric} tokens',
                f'{metric} Interpolation',
                f'{metric} Interpolation summed channels'
            ) ] + [f'{metric} Deviation']

    fig_plotly = make_subplots(
                rows=2 * figs_pr_sample + 1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=titles,
                )
        
    return figs_pr_sample,fig_plotly

def plot_metric_dev(res_gen, metric_name, fig_plotly, times_int_summed, metric_dev_fig_row):
    metric_dev = res_gen.ic_dev
    metric_dev_mean = metric_dev.mean().item()
    express_to_suplot(fig_plotly, px.line(x=times_int_summed, y=metric_dev, line_shape='hv', labels=dict(x="Time", y=f"{metric_name} Deviation", color="Channel")), row=metric_dev_fig_row, col=1)
    fig_plotly.layout.annotations[10]['text'] = f"{metric_name} Deviation, mean={metric_dev_mean}"

def get_colors(features, n_points, result):
    n_channels = len(features)
    colors_tok = np.broadcast_to(features[None], (n_points, n_channels)).flatten()
    colors_int = np.broadcast_to(features[None], result.ic_int.shape).flatten()
    return colors_tok,colors_int

def calculate_times(shift_to_time : dict, decoding_start : int, before: torch.LongTensor, result : ICRes):
    time_before = torch.sum(torch.tensor(
                    [shift_to_time[tok[3].item()] for tok in before]
                ), dim=0).item()
    middle_tokens_gen = result.tok[decoding_start:result.decoding_end-1]
    shifts_middle_gen = [shift_to_time[tok[3].item()] for tok in middle_tokens_gen]
    time_middle = torch.sum(torch.tensor(
                    shifts_middle_gen
                ), dim=0).item() + time_before
                
    if result.timepoints.dim() == 2:
        times_tok = np.array(result.timepoints + time_before ).flatten()
    else:
        raise NotImplementedError("This functionality has been deprecated")
        times_tok = np.broadcast_to(np.array(result.timepoints + time_before )[:,None], (n_points, n_channels)).flatten()

    times_int_summed = result.timepoints_int + time_before
                # NOTE: this should be handled differently
    times_int = np.broadcast_to(times_int_summed[:,None], result.ic_int.shape).flatten()
    inpaint_time = (time_before, time_middle)
    return inpaint_time, times_tok, times_int_summed, times_int

def plot_metric(metric, metric_name, fig_plotly, c_, times, metric_figure_row):
    scatter = px.scatter(
                        x=times,
                        y=metric,
                        color=c_,
                        # color_discrete_sequence=['red', 'green', 'blue'],
                        labels=dict(x="Time", y=metric_name, color="Channel"),
                        # color_continuous_scale="plasma",
                        )
    express_to_suplot(fig_plotly, scatter, row=metric_figure_row, col=1)

def plot_entr(fig_plotly, c_, times, entr, entr_figure_row):
    scatter = px.scatter(
                        x=times,
                        # y=result.entr_tok[:, channels].numpy().flatten(),
                        y=entr,
                        color=c_,
                        # color_discrete_sequence=['red', 'green', 'blue'],
                        labels=dict(x="Time", y="Entropy", color="Channel"),
                        # color_continuous_scale="plasma",
                        )
    express_to_suplot(fig_plotly, scatter, row=entr_figure_row, col=1)

def plot_piano_roll(fig_plotly, image_row, sr, piano_roll, inpaint_time):
    time_before, time_middle = inpaint_time
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

    express_to_suplot(fig_plotly, image, row=image_row, col=1)
    fig_plotly.add_shape(
                    go.layout.Shape(
                        type="line",
                        yref="paper",
                        xref="x",
                        x0=time_before,
                        y0=0,
                        x1=time_before,
                        y1=128,),
                    row=image_row,
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
                    row=image_row,
                    col=1
                )
    fig_plotly.update_yaxes(range=pitch_range, row=image_row, col=1)