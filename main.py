"""
@author: Gaetan Hadjeres
"""
from CIA.utils import get_free_port
from CIA.handlers import DecoderPrefixHandler
from CIA.positional_embeddings.positional_embedding import PositionalEmbedding
import importlib
import os
import shutil
from datetime import datetime
import time
import click
import torch

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from CIA.data_processors.data_processor import DataProcessor
from CIA.getters import (
    get_dataloader_generator,
    get_data_processor,
    get_decoder,
    get_handler,
    get_positional_embedding,
    get_sos_embedding,
)


@click.command()
@click.option("-t", "--train", is_flag=True)
@click.option("-l", "--load", is_flag=True)
@click.option("-o", "--overfitted", is_flag=True)
@click.option("-c", "--config", type=click.Path(exists=True))
@click.option("-n", "--num_workers", type=int, default=0)
def launcher(train, load, overfitted, config, num_workers):
    # === Init process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    # === Set shared parameters

    # always use the maximum number of available GPUs for training
    if train:
        world_size = torch.cuda.device_count()
        assert world_size > 0
    else:
        # only use 1 GPU for inference
        world_size = 1

    # Load config as dict
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace("/", ".")
    config = importlib.import_module(config_module_name).config

    # Compute time stamp
    if config["timestamp"] is not None:
        timestamp = config["timestamp"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        config["timestamp"] = timestamp

    # Create or retreive model_dir
    if load:
        model_dir = os.path.dirname(config_path)
    else:
        model_dir = f'models/{config["savename"]}_{timestamp}'

    # Copy .py config file in the save directory before training
    if not load:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        shutil.copy(config_path, f"{model_dir}/config.py")

    print(f"Using {world_size} GPUs")
    mp.spawn(
        main,
        args=(train, load, overfitted, config, num_workers, world_size, model_dir),
        nprocs=world_size,
        join=True,
    )


def main(rank, train, load, overfitted, config, num_workers, world_size, model_dir):
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

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
        training_phase=train,
        handler_type=config["handler_type"],
    )

    decoder.to(device)
    decoder = DistributedDataParallel(
        module=decoder,
        device_ids=[rank],
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

    if train:
        decoder_handler.train_model(
            batch_size=config["batch_size"],
            num_batches=config["num_batches"],
            num_epochs=config["num_epochs"],
            lr=config["lr"],
            plot=True,
            num_workers=num_workers,
        )
        exit()

    # fix projection matrices before generating
    if hasattr(decoder_handler.model.module.transformer, "fix_projection_matrices_"):
        decoder_handler.model.module.transformer.fix_projection_matrices_()

    exemple = dict(
        path="/home/leo/Code/CIA/examples/piano_sofiane.mid",
        num_events_middle=40,
        start=0,
    )
    # exemple = None

    # null_superconditioning
    # null_superconditioning = None
    null_superconditioning = [1, 1.5, 2, 2.5]

    if exemple is None:  # use dataloader
        (_, generator_val, _) = dataloader_generator.dataloaders(
            batch_size=1, num_workers=num_workers, shuffle_val=True
        )
        original_x = next(generator_val)["x"]
        x, metadata_dict = data_processor.preprocess(original_x, num_events_middle=None)
    else:
        # read midi file
        x = dataloader_generator.dataset.process_score(exemple["path"])
        # add pad, start and end symbols
        x = dataloader_generator.dataset.add_start_end_symbols(
            x,
            start_time=exemple["start"],
            sequence_size=dataloader_generator.sequences_size,
        )
        # tokenize
        x = dataloader_generator.dataset.tokenize(x)
        # to torch tensor
        original_x = torch.stack(
            [torch.tensor(x[e]).long() for e in dataloader_generator.features], dim=-1
        ).unsqueeze(0)
        # preprocess
        x, metadata_dict = data_processor.preprocess(
            original_x, num_events_middle=exemple["num_events_middle"]
        )

    # reconstruct original sequence to check post-processing
    # x_postprocess = data_processor.postprocess(
    #     x, decoding_end=metadata_dict['decoding_end'], metadata_dict=metadata_dict)

    ############################################################
    # inpainting
    # start_time = time.time()
    # x_gen, generated_region, decoding_end, num_event_generated, done = decoder_handler.inpaint(
    #     x=x.clone(), metadata_dict=metadata_dict, temperature=1., top_p=0.95, top_k=0)
    # end_time = time.time()
    ############################################################
    start_time = time.time()
    (
        x_gen,
        generated_region,
        decoding_end,
        num_event_generated,
        done,
    ) = decoder_handler.inpaint_non_optimized(
        x=x.clone(), metadata_dict=metadata_dict, temperature=1.0, top_p=0.95, top_k=0
    )
    end_time = time.time()
    ############################################################
    x_inpainted = data_processor.postprocess(x_gen, decoding_end, metadata_dict)
    # Timing infos
    if type(num_event_generated) == int:
        num_event_generated = [num_event_generated] * len(x)
    print(f"Num events_generated: {[e for e in num_event_generated]}")
    print(f"Time generation: {(end_time - start_time) / len(num_event_generated)}")
    average_time_list = [(end_time - start_time) / e for e in num_event_generated]
    print(
        f"Average time per generated event: {sum(average_time_list) / float(len(average_time_list))}"
    )

    # Saving
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if not os.path.exists(f"{decoder_handler.model_dir}/generations"):
        os.mkdir(f"{decoder_handler.model_dir}/generations")
    for k, tensor_score in enumerate(x_inpainted):
        if null_superconditioning is not None:
            path_no_extension = f"{decoder_handler.model_dir}/generations/{timestamp}_null{null_superconditioning[k]}"
        else:
            path_no_extension = (
                f"{decoder_handler.model_dir}/generations/{timestamp}_{k}"
            )
        decoder_handler.dataloader_generator.write(tensor_score, path_no_extension)
    for k, tensor_score in enumerate(original_x):
        path_no_extension = (
            f"{decoder_handler.model_dir}/generations/{timestamp}_{k}_original"
        )
        decoder_handler.dataloader_generator.write(tensor_score, path_no_extension)
    # for k, tensor_score in enumerate(x_postprocess):
    #     path_no_extension = f'{decoder_handler.model_dir}/generations/{timestamp}_{k}_original_postprocess'
    #     decoder_handler.dataloader_generator.write(tensor_score,
    #                                                path_no_extension)


if __name__ == "__main__":
    launcher()
