import copy
from pathlib import Path
from typing import List
import einops
import torch
import os

import tqdm
from CIA.dataloaders.piano_dataloader import PianoDataloaderGenerator
from CIA.dataset_managers.piano_midi_dataset import PAD_SYMBOL, PianoMidiDataset
from CIA.getters import get_handler, get_data_processor, \
    get_dataloader_generator, get_decoder, get_positional_embedding,\
    get_sos_embedding
import time
import importlib
from CIA.ic import DrawnICCurve, ICRes, Interpolator
from CIA.positional_embeddings.positional_embedding import PositionalEmbedding
from torch.nn.parallel import DistributedDataParallel
from CIA.utils import get_free_port
from CIA.data_processors.data_processor import DataProcessor
import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
import multiprocessing
from ic.app import Config
from ic.eval_ import eval_
from ic.plot import plot

model_dir = 'models/piano_event_performer_2021-10-01_16:03:06'
# logging.getLogger()


def gen(c : Config, device='cpu'):
    logger = multiprocessing.get_logger()
    # config =  importlib.import_module('CIA.configs.piarceiverStack').config
    # NOTE: override configuration
    dataloader_generator, data_processor, decoder_handler = load_pia(device)
    # NOTE: parallelize over the pieces 
    # for piece in tqdm.tqdm(c.pieces, desc='Pieces completed', disable=rank != 0, position=0):
    # TODO: this is quite ugly, but how can it be done prettyer using jsonargparse?
    c.experiment.dataset.dataloader_generator = dataloader_generator
    for original_x, piece_name, n_inpaint, end_window, piece in tqdm.tqdm(c.experiment.dataset, desc='Pieces completed', disable=rank != 0, position=0, leave=True):
        piece_folder = c.out.joinpath(piece_name)
        piece_folder.mkdir(exist_ok=True, parents=True)
        
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
            

        timepoints_tok_template, template_inpaint_end, ic_tok_template, entr_tok_template = decoder_handler.compute_token_onsets(
            x=metadata_dict['original_sequence'],
            metadata_dict=metadata_dict,
            onset_on_next_note = c.experiment.onset_on_next_note
        )
        # TODO: Should not necessarily be the case, but for now we assign events to the first timepoint
        if c.experiment.match_metric == 'ic':
            metric_tok_template = ic_tok_template
        elif c.experiment.match_metric == 'typicality':
            metric_tok_template = entr_tok_template - ic_tok_template
        # warn('Most likely, we would actually need to start sampling with a shift, if first note should not always align.')
        # TODO: 'Most likely, we would actually need to start sampling with a shift, if first note should not always align.'
        interpolation_template = Interpolator(
            metric_times=[timepoints_tok_template],
            metric=[metric_tok_template],
            weight_fn=c.experiment.weight,
            metric_clip = c.experiment.metric_clip_,
            reduce_equal_times = c.experiment.reduce_equal_times,
        )
        if c.experiment.ic_curve is None:
            ic_match_curve = interpolation_template
            timepoints_tok_match_curve = timepoints_tok_template
            ic_tok_match_curve = ic_tok_template
            entr_tok_match_curve = entr_tok_template
            match_curve_inpaint_end =  template_inpaint_end 
        else:
            ic_match_curve = c.experiment.ic_curve
            timepoints_tok_match_curve = None
            ic_tok_match_curve = None
            entr_tok_match_curve = None
            match_curve_inpaint_end = placeholder_duration
        
        for i in tqdm.tqdm(np.arange(rank, c.samples_per_template, world_size), desc='sample pr piece', disable=rank != 0, position=1, leave=False):
            gen_folder = piece_folder.joinpath(f'{i}')
            gen_folder.mkdir(exist_ok=True, parents=True)
            gen_file = gen_folder.joinpath(f'ic.pt')
            if gen_file.exists():
                continue
            # NOTE: might not be necessary to clone/copy
            x = x.clone()
            metadata_dict = copy.deepcopy(metadata_dict)
            # ts = c.experiment.time_points_generator(x, metadata_dict)
            # TODO: jsonargparse hack since placeholder_duration is not available at initialization time.
            c.experiment.time_points_generator.initialize(placeholder_duration)

            # TODO: ugly to check like this. Alternatively we could require that
            # the times are always relative. However, would be problematic for matching,
            # since the absolute time differences are important for the scaling.
            if isinstance(c.experiment.ic_curve, DrawnICCurve):
                ic_curve : DrawnICCurve = c.experiment.ic_curve
                ic_curve.set_placeholder_length(placeholder_duration)
            original_sequence = x.detach().cpu().clone()
            gen = decoder_handler.inpaint_ic_curve(
                x=x.clone(),
                sampling_config=c.sampling_config,
                experiment=c.experiment,
                metadata_dict=metadata_dict,
                piece=piece,
                num_max_generated_events=None,
                interpolator_template=ic_match_curve,
            )
            interpolation_time_points = gen.timepoints_int.clone()
            ic_int_temp = ic_match_curve(interpolation_time_points[None])[0]

            temp = ICRes(
                tok = original_sequence[0],
                ic_tok = ic_tok_match_curve,
                entr_tok = entr_tok_match_curve,
                timepoints = timepoints_tok_match_curve,
                ic_int = ic_int_temp,
                timepoints_int = interpolation_time_points,
                decoding_end=metadata_dict["decoding_end"].item(),
                piece = piece,
                inpaint_end = match_curve_inpaint_end
            )

            
            template_folder = piece_folder.joinpath('temp')
            template_folder.mkdir(exist_ok=True, parents=True)

            before, after = post_process_temp(
                dataloader_generator, 
                data_processor,
                template_folder,
                temp,
                write_template= i == 0 and rank == 0
            )
            decoding_start = data_processor.num_events_after+data_processor.num_events_before+2
    
            post_process_gen(dataloader_generator, before, after, gen_folder, gen, decoding_start)
    del decoder_handler
    decoder_handler = None
    torch.cuda.empty_cache()
def post_process_temp(
        dataloader_generator : PianoDataloaderGenerator,
        data_processor : DataProcessor,
        file_folder : Path,
        res_temp : ICRes,
        write_template = True
    ):
        decoding_start = data_processor.num_events_after + data_processor.num_events_before+2
        ds : PianoMidiDataset = dataloader_generator.dataset
        num_middle_tokens = res_temp.decoding_end - 1 - (data_processor.num_events_after+data_processor.num_events_before+2)
        if res_temp.piece is not None:
            sequence = ds.process_score(res_temp.piece.path)
            sequence = ds.tokenize(sequence)
            sequence = torch.tensor([sequence[e] for e in dataloader_generator.features])
            sequence = einops.rearrange(sequence, 'f n -> n f')
            before = sequence[:data_processor.num_events_before+res_temp.piece.start_node]
            after = sequence[res_temp.piece.start_node+data_processor.num_events_before+num_middle_tokens:]
        else:
            before = res_temp.tok[:data_processor.num_events_before]
            after = res_temp.tok[data_processor.num_events_before+1: decoding_start-1]
            middle_tokens_temp = res_temp.tok[decoding_start:res_temp.decoding_end-1]
            sequence = torch.cat([before, middle_tokens_temp, after], axis=0)
        shift_to_time = dataloader_generator.dataset.index2value['time_shift']
        # NOTE: disable the padding tokens 
        time_before = torch.sum(torch.tensor(
                [shift_to_time[tok[3].item()] for tok in before if isinstance(shift_to_time[tok[3].item()], float)]
            ), dim=0).item()
        if res_temp.timepoints is not None:
            res_temp.timepoints += time_before
        res_temp.timepoints_int += time_before
        res_temp.inpaint_end += time_before
        
        if write_template:
            res_temp.write(file_folder.joinpath(f'ic.pt'))
            temp_midi = file_folder.joinpath(f'song.mid')
            dataloader_generator.write(sequence, file_folder.joinpath(temp_midi.stem))
        return before, after
def post_process_gen(
        dataloader_generator : PianoDataloaderGenerator,
        before : torch.Tensor,
        after : torch.Tensor,
        gen_folder : Path,
        res_gen : ICRes,
        decoding_start : int
    ):
    middle_tokens_gen = res_gen.tok[decoding_start:res_gen.decoding_end-1]
    s = torch.cat([before, middle_tokens_gen, after], axis=0)
    gen_midi = gen_folder.joinpath(f'song.mid')
    dataloader_generator.write(s, gen_midi.parent.joinpath(gen_midi.stem))
    shift_to_time = dataloader_generator.dataset.index2value['time_shift']
    time_before = torch.sum(torch.tensor(
        [shift_to_time[tok[3].item()] for tok in before if isinstance(shift_to_time[tok[3].item()], float)]
    ), dim=0).item()
    res_gen.timepoints += time_before
    res_gen.timepoints_int += time_before
    res_gen.inpaint_end += time_before
    res_gen.write(gen_folder.joinpath(f'ic.pt'))

def load_pia(device, skip_model=False):
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
    if skip_model:
        decoder_handler = None
    else:
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
        
    return dataloader_generator,data_processor,decoder_handler
            # except Exception as e:
            #     logger.error(f"Failed generating sample {piece_name} with expection {e}")

        # x_inpainted = torch.cat([before, generated_region, after, end], axis=1)
        # x_inpainted = data_processor.postprocess(x_gen, decoding_end, metadata_dict)
        # x_inpainted_orig = data_processor.postprocess(original_x, decoding_end, metadata_dict)
        # print(f"Time of generated sequence {dataloader_generator.get_elapsed_time(x_inpainted[0][None])[0,-1]}")
        # dataloader_generator.write(x_inpainted_orig[0], 'out/orig')
                # fig_plotly.write_image(sample.parent.joinpath('ic_curve.svg'))
            # except Exception as e:
            #     logger = multiprocessing.get_logger()
            #     logger.error(f"Failed with expection {e}")

if __name__ == "__main__":
    parser = ArgumentParser(
        parser_mode="omegaconf"
        # default_config_files=['configs/config.yaml']
    )
    # parser.add_argument("--app", type=Config, nargs='*')  
    parser.add_argument("--app", type=List[Config])  
    parser.add_argument("--config", action=ActionConfigFile)
    # NOTE: needed for torch.distributed.launch
    parser.add_argument("--local_rank", type=int, default=None)
    subcommands = parser.add_subcommands()
    gen_subcomm = ArgumentParser()
    subcommands.add_subcommand("gen", gen_subcomm)
    plot_subcomm = ArgumentParser()
    subcommands.add_subcommand("plot", plot_subcomm)
    eval_subcomm = ArgumentParser()
    eval_subcomm.add_argument("--out_file", type=str, default='out/results/result.h5')
    subcommands.add_subcommand("eval", eval_subcomm)
    args = parser.parse_args()
    init = parser.instantiate_classes(args)
    app : List[Config] = init.app
    if args.subcommand == 'gen':
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
        for i, c in enumerate(app):
            # TODO: should be moved to top, but it's defined at the config level, and not the app level
            if c.seed is not None:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                seed = c.seed+rank
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            if 'RANK' not in  os.environ or int(os.environ['RANK']) == 0 :
                dir = Path(c.out)
                dir.mkdir(exist_ok=True, parents=True)
                args_exp = copy.copy(args)
                args_exp.app = [args.app[i]]
                parser.save(args_exp, dir.joinpath('config.yaml'), overwrite=True)
            print(f'Experiment: {c.experiment}\n Sampling Config: {c.sampling_config}, folder: {c.out}')
            gen(c, device=device)
            # NOTE: allow the processes to finish before plotting
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if 'RANK' not in os.environ or int(os.environ['RANK']) == 0 :    
                plot(c)

    elif args.subcommand == "plot":
        # dataloader_generator,data_processor,decoder_handler = load_pia(device='cpu', skip_model=True)
        for c in app:
            plot(c)
    elif args.subcommand == "eval":
        eval_(app, init.eval.out_file)
    else:
        raise ValueError(f"Unknown subcommand {args.subcommand}")

