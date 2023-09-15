from CIA.getters import get_data_processing, get_model, get_positional_embedding, get_sos_embedding
import importlib

from CIA.positional_embeddings.positional_embedding import PositionalEmbedding
config =  importlib.import_module('CIA.configs.piarceiverRw').config
dataloader_generator, data_processor = get_data_processing(
    dataset=config["dataset"],
    dataloader_generator_kwargs=config["dataloader_generator_kwargs"],
    data_processor_kwargs=config["data_processor_kwargs"],
)
