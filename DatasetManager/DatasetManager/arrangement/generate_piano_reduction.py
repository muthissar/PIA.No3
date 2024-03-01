# """
# Generate the piano reduction of orchestral xml files using a pre-trained model
# """
# import glob
# import os
# import re
# import shutil
#
# import numpy as np
# import torch
#
# from DatasetManager.arrangement.arrangement_helper import OrchestraIteratorGenerator
# from DatasetManager.dataset_manager import DatasetManager
#
# # Ugliness to the max, but I don't want to add Transformer to the venv path
# import sys
#
# config = get_config()
# dataset_manager_path = os.path.dirname(os.path.realpath(DatasetManager.__file__))
# database_path = f'{dataset_manager_path}/databases'
# dump_folder = f'{dataset_manager_path}/dump'
# transformer_path = f'{dataset_manager_path}/arranag'
# sys.path.append(config["transformer_path"])
# from Transformer.reduction.reduc_data_processor import ReductionDataProcessor
# from Transformer.transformer import Transformer
#
#
# class Reducter:
#     def __init__(self,
#                  writing_dir,
#                  corpus_it_gen,
#                  subdivision_model=2,
#                  subdivision_read=4,
#                  sequence_size=3,
#                  velocity_quantization=2,
#                  temperature=1.2):
#         """
#         :param subdivision: number of sixteenth notes per beat
#         """
#         self.subdivision_read = subdivision_read
#         self.sequence_size = sequence_size
#         self.velocity_quantization = velocity_quantization
#         self.writing_dir = writing_dir
#
#         #################################################################
#         #  Need the old db used to train the model (yes it sucks...)
#         dataset_manager = DatasetManager()
#         arrangement_dataset_kwargs = {
#             'transpose_to_sounding_pitch': True,
#             'subdivision': subdivision_model,
#             'sequence_size': sequence_size,
#             'velocity_quantization': velocity_quantization,
#             'max_transposition': 12,
#             'compute_statistics_flag': False
#         }
#         dataset = dataset_manager.get_dataset(
#             name='arrangement_large',
#             **arrangement_dataset_kwargs
#         )
#
#         #  Model params (need to know them :))
#         num_heads = 8
#         per_head_dim = 64
#         local_position_embedding_dim = 8
#         position_ff_dim = 1024
#         hierarchical = False
#         block_attention = False
#         nade = False
#         conditioning = True
#         double_conditioning = False
#         num_layers = 2
#         suffix = 'TEST'
#
#         reducer_input_dim = num_heads * per_head_dim
#
#         processor_encoder = ReductionDataProcessor(dataset=dataset,
#                                                    embedding_dim=reducer_input_dim - local_position_embedding_dim,
#                                                    reducer_input_dim=reducer_input_dim,
#                                                    local_position_embedding_dim=local_position_embedding_dim,
#                                                    flag='orchestra',
#                                                    block_attention=False)
#
#         processor_decoder = ReductionDataProcessor(dataset=dataset,
#                                                    embedding_dim=reducer_input_dim - local_position_embedding_dim,
#                                                    reducer_input_dim=reducer_input_dim,
#                                                    local_position_embedding_dim=local_position_embedding_dim,
#                                                    flag='piano',
#                                                    block_attention=block_attention)
#
#         processor_encodencoder = None
#         #################################################################
#
#         #################################################################
#         # Init model
#         # Use all gpus available
#         gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
#         print(gpu_ids)
#
#         self.model = Transformer(
#             dataset=dataset,
#             data_processor_encodencoder=processor_encodencoder,
#             data_processor_encoder=processor_encoder,
#             data_processor_decoder=processor_decoder,
#             num_heads=num_heads,
#             per_head_dim=per_head_dim,
#             position_ff_dim=position_ff_dim,
#             hierarchical_encoding=hierarchical,
#             block_attention=block_attention,
#             nade=nade,
#             conditioning=conditioning,
#             double_conditioning=double_conditioning,
#             num_layers=num_layers,
#             dropout=0.1,
#             input_dropout=0.2,
#             reduction_flag=True,
#             lr=1e-4,
#             gpu_ids=gpu_ids,
#             suffix=suffix
#         )
#         #################################################################
#
#         self.corpus_it_gen = corpus_it_gen
#
#         self.temperature = temperature
#
#         return
#
#     def iterator_gen(self):
#         return (score for score in self.corpus_it_gen())
#
#     def __call__(self, model_path):
#         #  Load model weights
#         self.model.load_overfit(model_path)
#
#         for arr_pair in self.iterator_gen():
#             filepath = arr_pair['Orchestra']
#
#             context_size = self.model.data_processor_decoder.num_frames_piano - 1
#
#             #  Load input piano score
#             piano_init, rhythm_orchestra, orchestra = \
#                 self.model.data_processor_encoder.init_reduction_filepath(batch_size=1,
#                                                                           filepath=filepath,
#                                                                           subdivision=self.subdivision_read)
#
#             piano = self.model.generation_reduction(
#                 piano_init=piano_init,
#                 orchestra=orchestra,
#                 temperature=self.temperature,
#                 batch_size=1,
#                 plot_attentions=False
#             )
#
#             piano_cpu = piano[:, context_size:-context_size].cpu()
#             # Last duration will be a quarter length
#             duration_piano = np.asarray(list(rhythm_orchestra[1:]) + [self.subdivision_read]) - np.asarray(
#                 list(rhythm_orchestra[:-1]) + [0])
#
#             generated_piano_score = self.model.dataset.piano_tensor_to_score(piano_cpu,
#                                                                              durations=duration_piano,
#                                                                              subdivision=self.subdivision_read)
#
#             # Copy the whole folder to writing dir
#             src_dir = os.path.dirname(filepath)
#             shutil.copytree(src_dir, self.writing_dir)
#             # Remove old (midi) files
#             new_midi_path = re.sub(src_dir, self.writing_dir, filepath)
#             os.remove(new_midi_path)
#             # Write generated piano score and orchestra in xml
#             generated_piano_score.write(fp=f"{self.writing_dir}/{filepath}_piano.xml", fmt='musicxml')
#             orchestra.write(fp=f"{self.writing_dir}/{filepath}_orch.xml", fmt='musicxml')
#
#         return
#
#
# def prepare_db(root_dir, write_dir):
#     """
#     Basically remove the hiearchical structure of kunst by appending composer names to folders
#     :return:
#     """
#     dirs = glob.glob(f'{root_dir}/*/*')
#     for dir in dirs:
#         dir_split = re.split('/', dir)
#         composer_name = dir_split[-2]
#         piece_name = dir_split[-1]
#         new_name = f'{write_dir}/{composer_name}_{piece_name}'
#         shutil.copytree(dir, new_name)
#     return
#
#
# if __name__ == '__main__':
#     db_path = '/home/leo/Recherche/Databases/Orchestration/BACKUP/Kunstderfuge/Selected_works_clean'
#     writing_dir = '/home/leo/Recherche/Databases/Orchestration/arrangement_mxml/kunstderfuge'
#
#     # src = '/home/leo/Recherche/Databases/Orchestration/BACKUP/Kunstderfuge/Selected_works'
#     # dest = '/home/leo/Recherche/Databases/Orchestration/BACKUP/Kunstderfuge/Selected_works_clean'
#     # prepare_db(src, dest)
#
#     model_path = '/home/leo/Recherche/Code/Transformer/models_backup/Reducter-2_ArrangementDataset-arrangement_large-2-3-2-12/'
#
#     orchestra_iterator = OrchestraIteratorGenerator(
#         folder_path=db_path,
#         process_file=False
#     )
#
#     reducter = Reducter(
#         writing_dir,
#         orchestra_iterator,
#         subdivision_model=2,
#         subdivision_read=4,
#         sequence_size=3,
#         velocity_quantization=2,
#         temperature=1.2)
#
#     reducter(model_path)
