from pathlib import Path
from bs4 import BeautifulSoup
from CIA.ic import DataPiece, ICRes
from ic.app import Config
import soundfile as sf
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pretty_midi
from plotly.subplots import make_subplots


from warnings import warn


def express_to_suplot(fig_plotly, explot, row, col):
    for trace in explot.data:
        trace.showlegend = False
        fig_plotly.add_trace(trace, row=row, col=col)


def plot(c : Config, sr : int = 25):
    # TODO: this is not pretty
    crop_start_time = 0
    crop_end_time = 150
    # TODO: change all naming to metric
    if not isinstance(c.experiment.dataset, DataPiece):
        warn('Check implimentation, since it is not clear how to handle this case')
    for temp_file in [f for f in c.out.rglob('*/temp/ic.pt')]:
        # NOTE: since right now we don't pass the dataprocessor, we cannot calculate the exact generated files. 
        # Therefore not to recompute everything plot only those which are not already plotted
        if temp_file.parent.joinpath(f'{c.experiment.match_metric}_curve.html').exists():
            continue
        song_dir = temp_file.parent.parent
        res_temp = ICRes.load(p=temp_file)
        temp_midi = temp_file.parent.joinpath(f'song.mid')
        for sample in song_dir.rglob('*/ic.pt'):
            if sample == temp_file:
                continue
            res_gen = ICRes.load(p=sample)
            gen_midi = sample.parent.joinpath(f'song.mid')

            midi_files = [temp_midi, gen_midi]
            results = [res_temp, res_gen]
            metric_name = c.experiment.match_metric
            metric_name = metric_name.capitalize()
            figs_pr_sample, fig_plotly = get_figure(metric_name)
            
            audio_files = []
            for i, (midi_file, result) in enumerate(zip(midi_files, results)):
                midi = pretty_midi.PrettyMIDI(str(midi_file))
                midi.instruments[0].notes = [n for n in midi.instruments[0].notes if n.start >= crop_start_time and n.start <= crop_end_time]
                audio_sr = 44100
                audio = midi.fluidsynth(fs=audio_sr)
                audio_file = midi_file.parent.joinpath(f'song.mp3')
                sf.write(file=audio_file, data=audio, samplerate=audio_sr)
                audio_files.append(Path('..').joinpath(audio_file.relative_to(audio_file.parent.parent)))
                # NOTE: calculate timepoints
                times_int_summed = result.timepoints_int.flatten()
                times_int = np.broadcast_to(times_int_summed[:,None], result.ic_int.shape)
                if result.timepoints is not None:
                    times_tok = result.timepoints.flatten()
                    n_toks = len(result.timepoints)
                else:
                    times_tok = None
                    n_toks = 0
                warn('hack, refactor this')
                inpaint_time = (0, 10)
                # inpaint_time = (times_tok[0], times_tok[-1])
                
                # NOTE: colors
                features = np.array(['pitch', 'velocity', 'duration', 'time_shift'])
                n_int_points  = result.ic_int.shape[0]
                colors_tok, colors_int = get_colors(features, n_toks, n_int_points)
                
                # NOTE: figure rows
                img_offset = i*figs_pr_sample
                piano_roll_fig_row = img_offset+1
                entr_fig_row = img_offset + 2
                metric_fig_row = img_offset + 3
                int_fig_row = img_offset + 4
                int_summed_fig_row = img_offset + 5

                metric_dev_fig_row = 1 *figs_pr_sample + 6
            
                # NOTE: plot piano roll
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
                express_to_suplot(fig_plotly, px.line(x=times_int.flatten(), y=result.ic_int.flatten(), color=colors_int, line_shape='hv', labels=dict(x="Time", y=metric_name, color="Channel")), row=int_fig_row, col=1)
                int_summed_channels = result.ic_int.sum(-1)
                express_to_suplot(fig_plotly, px.line(x=times_int_summed, y=int_summed_channels, line_shape='hv', labels=dict(x="Time", y=metric_name, color="Channel")), row=int_summed_fig_row, col=1)
            plot_metric_dev(res_gen, metric_name, fig_plotly, times_int_summed, metric_dev_fig_row)
            
            # NOTE update y-axis
            for i in range(2):
                metric_int_max = max([r.ic_int.max() for r in results])
                metric_int_summed_max = max([r.ic_int.sum(-1).max() for r in results])
                if results[0].ic_tok is not None:
                    entrs_max = max([r.entr_tok.max() for r in results])
                    cum_metric_max = max([r.ic_tok.max() for r in results])
                    fig_plotly.update_yaxes(range=(0, entrs_max+1), row=entr_fig_row, col=1)
                    fig_plotly.update_yaxes(range=(0, cum_metric_max+1), row=metric_fig_row, col=1)
                fig_plotly.update_yaxes(range=(0, metric_int_max+1), row=int_fig_row, col=1)
                fig_plotly.update_yaxes(range=(0, metric_int_summed_max+1), row=int_summed_fig_row, col=1)
            fig_height = 2000
            fig_plotly.update_xaxes(tickformat="%M:%S.%3f")
            fig_plotly.update_layout(height=fig_height, xaxis=dict(range=[crop_start_time, crop_end_time]))
            html_file = sample.parent.joinpath(f'{metric_name}_curve.html')
            html = fig_plotly.to_html()
            soup = BeautifulSoup(html, features="html.parser")
            audio_elems = [soup.new_tag('audio', controls=True, src=audio_file) for audio_file in audio_files]
            soup.div.insert_before(*audio_elems)
            with open(html_file, 'w') as file:
                file.write(soup.prettify())
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

def plot_metric_dev(res_gen : ICRes, metric_name, fig_plotly, times_int_summed, metric_dev_fig_row):
    metric_dev = res_gen.ic_dev
    metric_dev_mean = metric_dev.mean().item()
    express_to_suplot(fig_plotly, px.line(x=times_int_summed, y=metric_dev, line_shape='hv', labels=dict(x="Time", y=f"{metric_name} Deviation", color="Channel")), row=metric_dev_fig_row, col=1)
    fig_plotly.layout.annotations[10]['text'] = f"{metric_name} Deviation, mean={metric_dev_mean}"

def get_colors(features, n_toks, n_int_points):
    n_channels = len(features)
    colors_tok = np.broadcast_to(features[None], (n_toks, n_channels)).flatten()
    colors_int = np.broadcast_to(features[None], (n_int_points, n_channels)).flatten()
    return colors_tok,colors_int


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
                    x=np.arange(0, end, 1/sr)[:piano_roll.shape[0]],
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
    # fig_plotly.update_xaxes(hoverformat="%M:%S.%3f", row=image_row, col=1)