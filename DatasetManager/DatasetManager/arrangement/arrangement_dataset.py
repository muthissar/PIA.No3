import os
import os
import re
import shutil

import matplotlib.pyplot as plt
import music21
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

import DatasetManager.arrangement.nw_align as nw_align
from DatasetManager.arrangement.arrangement_helper import quantize_velocity_pianoroll_frame, unquantize_velocity, \
    shift_pr_along_pitch_axis, score_to_pianoroll, flatten_dict_pr, new_events, pitch_class_matrix
from DatasetManager.helpers import REST_SYMBOL, SLUR_SYMBOL, END_SYMBOL, START_SYMBOL, \
    YES_SYMBOL, NO_SYMBOL, PAD_SYMBOL
from DatasetManager.orchestration.orchestration_dataset import OrchestrationDataset

"""
Piano at time t is a vector
[pitch_0, pitch_1, ... , pitch_N]
with each pitch_i in [0,V] where V is the number of discrete velocities. 
Usually I use V = 1 and pitch_i simply indicates if a note is played or not 
"""


class ArrangementDataset(OrchestrationDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 corpus_it_gen_instru_range,
                 name,
                 subdivision,
                 sequence_size,
                 velocity_quantization,
                 max_transposition,
                 integrate_discretization,
                 alignement_type,
                 transpose_to_sounding_pitch,
                 compute_statistics_flag):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super().__init__(corpus_it_gen,
                         name,
                         subdivision,
                         sequence_size,
                         velocity_quantization,
                         max_transposition,
                         integrate_discretization,
                         transpose_to_sounding_pitch,
                         compute_statistics_flag
                         )

        self.alignement_type = alignement_type

        self.statistic_folder = f'{self.dump_folder}/arrangement/statistics'
        if os.path.isdir(self.statistic_folder):
            shutil.rmtree(self.statistic_folder)
        os.makedirs(self.statistic_folder)

        # mapping piano
        self.midi_pitch2index_piano = {}
        self.index2midi_pitch_piano = {}
        self.value2oneHot_perPianoToken = {}
        self.oneHot2value_perPianoToken = {}
        self.number_pitch_piano = None

        # Often used vectors, computed in compute_index_dicts
        self.precomputed_vectors_piano = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            PAD_SYMBOL: None,
            REST_SYMBOL: None,
        }
        return

    def __repr__(self):
        return f'ArrangementDataset-' \
               f'{self.name}-' \
               f'{self.subdivision}-' \
               f'{self.sequence_size}-' \
               f'{self.velocity_quantization}-' \
               f'{self.max_transposition}'

    def compute_index_dicts_piano(self, set_midiPitch_per_instrument):
        ############################################################
        ############################################################
        # Piano
        min_pitch_piano = min(set_midiPitch_per_instrument["Piano"])
        max_pitch_piano = max(set_midiPitch_per_instrument["Piano"])
        #  Use range to avoid "gaps" in the piano tessitura
        list_midiPitch = sorted(list(range(min_pitch_piano, max_pitch_piano + 1)))
        self.number_pitch_piano = len(list_midiPitch)
        for index, midi_pitch in enumerate(list_midiPitch):
            self.midi_pitch2index_piano[midi_pitch] = index
            self.index2midi_pitch_piano[index] = midi_pitch

        # One hot encoding for velocitites
        dict_for_velocity2oneHot = {}
        dict_for_oneHot2velocity = {}
        # Silence (start with silence mapped to zero, kinda more logical, and then velocity and oneHot are the same
        # value)
        index = 0
        dict_for_velocity2oneHot[REST_SYMBOL] = index
        dict_for_oneHot2velocity[index] = REST_SYMBOL
        for velocity in range(1, self.velocity_quantization):
            index += 1
            dict_for_velocity2oneHot[velocity] = index
            dict_for_oneHot2velocity[index] = velocity
        # Slur
        index += 1
        dict_for_velocity2oneHot[SLUR_SYMBOL] = index
        dict_for_oneHot2velocity[index] = SLUR_SYMBOL
        # Pad
        index += 1
        dict_for_velocity2oneHot[PAD_SYMBOL] = index
        dict_for_oneHot2velocity[index] = PAD_SYMBOL
        # Start
        index += 1
        dict_for_velocity2oneHot[START_SYMBOL] = index
        dict_for_oneHot2velocity[index] = START_SYMBOL
        # End
        index += 1
        dict_for_velocity2oneHot[END_SYMBOL] = index
        dict_for_oneHot2velocity[index] = END_SYMBOL

        for token_index, _ in self.index2midi_pitch_piano.items():
            self.value2oneHot_perPianoToken[token_index] = dict.copy(dict_for_velocity2oneHot)
            self.oneHot2value_perPianoToken[token_index] = dict.copy(dict_for_oneHot2velocity)
        ############################################################
        ############################################################

        ############################################################
        ############################################################
        # These are the one-hot representation of several useful (especially during generation) vectors
        piano_start_vector = []
        piano_end_vector = []
        piano_padding_vector = []
        piano_rest_vector = []
        for token_index, value2oneHot in self.value2oneHot_perPianoToken.items():
            piano_start_vector.append(value2oneHot[START_SYMBOL])
            piano_end_vector.append(value2oneHot[END_SYMBOL])
            piano_padding_vector.append(value2oneHot[PAD_SYMBOL])
            piano_rest_vector.append(value2oneHot[REST_SYMBOL])
        self.precomputed_vectors_piano[START_SYMBOL] = torch.from_numpy(np.asarray(piano_start_vector)).long()
        self.precomputed_vectors_piano[END_SYMBOL] = torch.from_numpy(np.asarray(piano_end_vector)).long()
        self.precomputed_vectors_piano[PAD_SYMBOL] = torch.from_numpy(np.asarray(piano_padding_vector)).long()
        self.precomputed_vectors_piano[REST_SYMBOL] = torch.from_numpy(np.asarray(piano_rest_vector)).long()
        return

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """

        print('Loading index dictionnary')

        self.load_index_dicts()

        print('Making tensor dataset')

        total_chunk_counter = 0
        too_many_instruments_frame = 0
        impossible_transposition = 0

        # Variables for statistics
        if self.compute_statistics_flag:
            scores = []
            num_frames_with_different_pitch_class = 0
            total_frames_counter = 0
            open(f"{self.statistic_folder}/different_set_pc.txt", 'w').close()

        # List storing piano and orchestra datasets
        piano_tensor_dataset = []
        orchestra_tensor_dataset = []
        orchestra_instruments_presence_tensor_dataset = []

        # Iterate over files
        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

            ############################################################
            # Compute original pianorolls
            pianoroll_piano, onsets_piano, _ = score_to_pianoroll(
                score=arr_pair['Piano'],
                subdivision=self.subdivision,
                simplify_instrumentation=None,
                instrument_grouping=self.instrument_grouping,
                transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
                integrate_discretization=self.integrate_discretization,
                binarize=False,
            )

            pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(
                score=arr_pair['Orchestra'],
                subdivision=self.subdivision,
                simplify_instrumentation=self.simplify_instrumentation,
                instrument_grouping=self.instrument_grouping,
                transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
                integrate_discretization=self.integrate_discretization,
                binarize=False,
            )

            ############################################################
            #  Align (we can use non transposed scores, changes nothing to the alignement
            corresponding_frames = self.align_score(piano_pr=pianoroll_piano,
                                                    piano_onsets=onsets_piano,
                                                    orchestra_pr=pianoroll_orchestra,
                                                    orchestra_onsets=onsets_orchestra)
            if self.compute_statistics_flag:
                scores.extend(this_scores)
            # Get corresponding pitch_classes (for statistics)
            pc_piano_list = [e[0][1] for e in corresponding_frames]
            pc_orchestra_list = [e[1][1] for e in corresponding_frames]

            ############################################################
            # Prepare chunks of indices
            chunks_piano_indices, chunks_orchestra_indices = self.prepare_chunk_from_corresponding_frames(
                corresponding_frames)

            # Quantize piano
            pr_piano = quantize_velocity_pianoroll_frame(pianoroll_piano["Piano"],
                                                         self.velocity_quantization)
            onsets_piano = onsets_piano["Piano"]

            pr_pair = {"Piano": pr_piano, "Orchestra": pianoroll_orchestra}
            onsets_pair = {"Piano": onsets_piano, "Orchestra": onsets_orchestra}

            # First get non transposed score
            transposition_semi_tone = 0
            minimum_transpositions_allowed = None
            maximum_transpositions_allowed = None
            minimum_transpositions_allowed, maximum_transpositions_allowed, \
            piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
            total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                self.transpose_loop_iteration(pr_pair, onsets_pair, transposition_semi_tone,
                                              chunks_piano_indices, chunks_orchestra_indices,
                                              minimum_transpositions_allowed, maximum_transpositions_allowed,
                                              piano_tensor_dataset, orchestra_tensor_dataset,
                                              orchestra_instruments_presence_tensor_dataset,
                                              total_chunk_counter, too_many_instruments_frame, impossible_transposition)

            for transposition_semi_tone in range(-self.max_transposition, self.max_transposition + 1):
                if transposition_semi_tone == 0:
                    continue
                _, _, piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                    self.transpose_loop_iteration(pr_pair, onsets_pair, transposition_semi_tone,
                                                  chunks_piano_indices, chunks_orchestra_indices,
                                                  minimum_transpositions_allowed, maximum_transpositions_allowed,
                                                  piano_tensor_dataset, orchestra_tensor_dataset,
                                                  orchestra_instruments_presence_tensor_dataset,
                                                  total_chunk_counter, too_many_instruments_frame,
                                                  impossible_transposition)

            if self.compute_statistics_flag:
                for pc_piano, pc_orchestra in zip(pc_piano_list, pc_orchestra_list):
                    total_frames_counter += 1
                    # Statistics: compare pitch class in orchestra and in piano
                    if pc_piano != pc_orchestra:
                        num_frames_with_different_pitch_class += 1
                        with open(f"{self.statistic_folder}/different_set_pc.txt", "a") as ff:
                            for this_pc in pc_piano:
                                ff.write(f"{this_pc} ")
                            ff.write("// ")
                            for this_pc in pc_orchestra:
                                ff.write(f"{this_pc} ")
                            ff.write("\n")

        piano_tensor_dataset = torch.cat(piano_tensor_dataset, 0)
        orchestra_tensor_dataset = torch.cat(orchestra_tensor_dataset, 0)
        orchestra_instruments_presence_tensor_dataset = torch.cat(orchestra_instruments_presence_tensor_dataset, 0)

        #######################
        if self.compute_statistics_flag:
            # NW statistics
            mean_score = np.mean(scores)
            variance_score = np.var(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            nw_statistics_folder = f"{self.statistic_folder}/nw"
            if os.path.isdir(nw_statistics_folder):
                shutil.rmtree(nw_statistics_folder)
            os.makedirs(nw_statistics_folder)
            with open(f"{nw_statistics_folder}/scores.txt", "w") as ff:
                ff.write(f"Mean score: {mean_score}\n")
                ff.write(f"Variance score: {variance_score}\n")
                ff.write(f"Max score: {max_score}\n")
                ff.write(f"Min score: {min_score}\n")
                for elem in scores:
                    ff.write(f"{elem}\n")
            # Histogram
            n, bins, patches = plt.hist(scores, 50)
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('Histogram NW scores')
            plt.savefig(f'{nw_statistics_folder}/histogram_score.pdf')

            # Pitch class statistics
            pitch_class_statistics_folder = f"{self.statistic_folder}/pitch_class"
            if os.path.isdir(pitch_class_statistics_folder):
                shutil.rmtree(pitch_class_statistics_folder)
            os.makedirs(pitch_class_statistics_folder)
            # Move different set pc
            shutil.move(f"{self.statistic_folder}/different_set_pc.txt",
                        f"{pitch_class_statistics_folder}/different_set_pc.txt")
            # Write the ratio of matching frames
            with open(f"{pitch_class_statistics_folder}/ratio_matching_pc_set.txt", "w") as ff:
                ff.write(f"Different PC sets: {num_frames_with_different_pitch_class}\n")
                ff.write(f"Total number frames: {total_frames_counter}\n")
                ff.write(f"Ratio: {num_frames_with_different_pitch_class / total_frames_counter}\n")
        #######################

        #######################
        #  Create Tensor Dataset
        dataset = TensorDataset(piano_tensor_dataset,
                                orchestra_tensor_dataset,
                                orchestra_instruments_presence_tensor_dataset)
        #######################

        print(
            f'### Sizes: \n'
            f'Piano: {piano_tensor_dataset.size()}\n'
            f'Orchestra: {orchestra_tensor_dataset.size()}\n'
            f'Chunks: {total_chunk_counter}\n'
            f'Too many instru chunks: {too_many_instruments_frame}\n'
            f'Impossible transpo: {impossible_transposition}')
        return dataset

    def get_score_tensor(self, scores, offsets):
        return

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return

    def get_metadata_tensor(self, score):
        return None

    def score_to_list_pc(self, pianoroll, onsets):
        flat_pr = flatten_dict_pr(pianoroll)

        #  Get new events indices (diff matrices)
        events = new_events(pianoroll, onsets)
        #  Pad at the end of the pitch axis to get a multiple of 12 (number of pitch classes)
        pr_event = np.pad(flat_pr[events], pad_width=[(0, 0), (0, 4)], mode='constant', constant_values=0)

        # Reduce on 12 pitch-classes
        length = len(events)
        # test = np.repeat(np.expand_dims(np.arange(0, 132), 0), length, axis=0)
        pcs = np.sum(np.reshape(pr_event, (length, 11, 12)), axis=1)

        list_pc = []
        for frame_index in range(len(events)):
            list_pc.append((events[frame_index], set(np.where(pcs[frame_index] > 0)[0])))

        return list_pc

    def compute_frames_distances(self, pr_piano, onsets_piano,
                                 pr_orchestra, onsets_orchestra):
        """
        Simultaneous notes = count
        +1 if its simultaneous onsets

        pitch class = count

        :param pr_piano:
        :param onsets_piano:
        :param pr_orchestra:
        :param onsets_orchestra:
        :return:
        """
        eps = 1e-10
        pr_orchestra_norm = pr_orchestra / (np.sum(pr_orchestra, axis=1, keepdims=True) + eps)
        onsets_orchestra_norm = onsets_orchestra / (np.sum(onsets_orchestra, axis=1, keepdims=True) + eps)

        pr_piano_pc = pitch_class_matrix(pr_piano, binarize=True)
        onsets_piano_pc = pitch_class_matrix(onsets_piano, binarize=True)
        pr_orchestra_pc = pitch_class_matrix(pr_orchestra, binarize=True)
        onsets_orchestra_pc = pitch_class_matrix(onsets_orchestra, binarize=True)
        pr_orchestra_pc_norm = pr_orchestra_pc / (np.sum(pr_orchestra_pc, axis=1, keepdims=True) + eps)
        onsets_orchestra_pc_norm = onsets_orchestra_pc / (np.sum(onsets_orchestra_pc, axis=1, keepdims=True) + eps)

        simultaneous_notes = np.matmul(pr_piano, pr_orchestra_norm.T)
        simultaneous_onsets = np.matmul(onsets_piano, onsets_orchestra_norm.T)
        simultaneous_notes_pc = np.matmul(pr_piano_pc, pr_orchestra_pc_norm.T)
        simultaneous_onsets_pc = np.matmul(onsets_piano_pc, onsets_orchestra_pc_norm.T)

        #  Silences
        silences_indices_piano = np.where(np.sum(pr_piano, axis=1) == 0)
        silences_indices_orchestra = np.where(np.sum(pr_orchestra, axis=1) == 0)
        silences_piano = np.zeros((len(pr_piano), 1))
        silences_orchestra = np.zeros((len(pr_orchestra), 1))
        silences_piano[silences_indices_piano] = 1
        silences_orchestra[silences_indices_orchestra] = 1
        silence_score = np.matmul(silences_piano, silences_orchestra.T)

        score = simultaneous_notes + simultaneous_onsets + \
                simultaneous_notes_pc + simultaneous_onsets_pc + \
                silence_score

        return score

    def align_score(self, piano_pr, piano_onsets, orchestra_pr, orchestra_onsets):

        #  1/ Pitch-class representation
        if self.alignement_type == 'pitch_class':
            list_pc_piano = self.score_to_list_pc(piano_pr, piano_onsets)
            list_pc_orchestra = self.score_to_list_pc(orchestra_pr, orchestra_onsets)
            alignement_input_piano = [e[1] for e in list_pc_piano]
            alignement_input_orchestra = [e[1] for e in list_pc_orchestra]

            corresponding_indices, score_matrix = nw_align.nwalign(alignement_input_piano, alignement_input_orchestra,
                                                                   score_matrix=None, gapOpen=-3, gapExtend=-1)
            corresponding_frames = [(list_pc_piano[ind_piano], list_pc_orchestra[ind_orchestra])
                                    for ind_piano, ind_orchestra in corresponding_indices]

            return corresponding_frames

        elif self.alignement_type == 'complete':
            #  2/ All-pitch, slurs and onsets

            # list_notes_piano = self.score_to_list_notes(piano_score, 'piano')
            # list_notes_orchestra = self.score_to_list_notes(orchestra_score, 'orchestra')
            # alignement_input_piano = [e[1] for e in list_notes_piano]
            # alignement_input_orchestra = [e[1] for e in list_notes_orchestra]

            pr_piano_event, onsets_piano_event, events_piano = self.score_to_list_notes_2(piano_pr, piano_onsets)
            pr_orchestra_event, onsets_orchestra_event, events_orchestra = self.score_to_list_notes_2(orchestra_pr,
                                                                                                      orchestra_onsets)
            matrix_frame_distances = self.compute_frames_distances(pr_piano_event, onsets_piano_event,
                                                                   pr_orchestra_event, onsets_orchestra_event)

            alignement_input_piano = list(range(len(pr_piano_event)))
            alignement_input_orchestra = list(range(len(pr_orchestra_event)))

            corresponding_indices, score_matrix = nw_align.nwalign(
                seqj=alignement_input_orchestra, seqi=alignement_input_piano,
                gapOpen=0, gapExtend=1, score_matrix=matrix_frame_distances)

            corresponding_frames = [((events_piano[e[1]], {}), (events_orchestra[e[0]], {})) for e in
                                    corresponding_indices]

            return corresponding_frames

    def score_to_list_notes_2(self, pianoroll, onsets):
        """
        [[(pitch, value), ..., ], ..., ]
        List of lists
        Each sublist is the list of notes played at this time
        Value indicate if the note is either off, onset, sustained
        Note that for the orchestra, no need to keep the instrument information, as it is not used for aligning
        :param score:
        :param datatype:
        :return:
        """
        #  Event representation
        events = new_events(pianoroll, onsets)
        pr_event = None
        onsets_event = None
        for _, matrix in pianoroll.items():
            if pr_event is None:
                pr_event = matrix[events]
            else:
                pr_event += matrix[events]
        for _, matrix in onsets.items():
            if onsets_event is None:
                onsets_event = matrix[events]
            else:
                onsets_event += matrix[events]

        return pr_event, onsets_event, events

    def get_allowed_transpositions_from_pr(self, pr, frames, instrument_name):
        #  Get min and max pitches
        pr_frames = np.asarray(
            [pr[frame] for frame in frames if
             frame not in [REST_SYMBOL, START_SYMBOL, END_SYMBOL, PAD_SYMBOL]])
        flat_pr = pr_frames.sum(axis=0)
        non_zeros_pitches = list(np.where(flat_pr > 0)[0])
        if len(non_zeros_pitches) > 0:
            min_pitch = min(non_zeros_pitches)
            max_pitch = max(non_zeros_pitches)

            # Compare with reference tessitura, and ensure min <= 0 and max >= 0
            allowed_transposition_down = min(0, self.observed_tessitura[instrument_name]["min"] - min_pitch)
            allowed_transposition_up = max(0, self.observed_tessitura[instrument_name]["max"] - max_pitch)
        else:
            allowed_transposition_down = None
            allowed_transposition_up = None

        return allowed_transposition_down, allowed_transposition_up

    def prepare_chunk_from_corresponding_frames(self, corresponding_frames):
        chunks_piano_indices = []
        chunks_orchestra_indices = []
        number_corresponding_frames = len(corresponding_frames)
        for index_frame in range(0, number_corresponding_frames):
            # if we consider the time in the middle is the one of interest, we must pad half of seq size at the
            # beginning and half at the end
            start_index = index_frame - (self.sequence_size - 1) // 2
            start_index_truncated = max(0, start_index)
            #  Always add at least one None frame at the beginning (instead of a the real previous frame)
            #  Hence, we avoid the model observe slurs from unseen previous frame
            padding_beginning = start_index_truncated - start_index
            end_index = index_frame + (self.sequence_size - 1) // 2
            end_index_truncated = min(number_corresponding_frames, end_index)
            padding_end = max(0, end_index - number_corresponding_frames + 1)

            #  Always include a None frame as first frame instead of the real previous frame.
            # This is because we don't want to have slurs from frames that the model cannot observe
            this_piano_chunk = [e[0][0] for e in corresponding_frames[start_index_truncated:end_index_truncated + 1]]
            this_orchestra_chunk = [e[1][0] for e in
                                    corresponding_frames[start_index_truncated:end_index_truncated + 1]]

            # Padding
            if padding_beginning == 0:
                prepend_vector = []
            else:
                prepend_vector = (padding_beginning - 1) * [PAD_SYMBOL] + [START_SYMBOL]

            if padding_end == 0:
                append_vector = []
            else:
                append_vector = [END_SYMBOL] + (padding_end - 1) * [PAD_SYMBOL]

            this_piano_chunk = prepend_vector + this_piano_chunk + append_vector
            this_orchestra_chunk = prepend_vector + this_orchestra_chunk + append_vector
            chunks_piano_indices.append(this_piano_chunk)
            chunks_orchestra_indices.append(this_orchestra_chunk)

        return chunks_piano_indices, chunks_orchestra_indices

    def transpose_loop_iteration(self, pianorolls_pair, onsets_pair, transposition_semi_tone,
                                 chunks_piano_indices, chunks_orchestra_indices,
                                 minimum_transposition_allowed, maximum_transposition_allowed,
                                 piano_tensor_dataset, orchestra_tensor_dataset,
                                 orchestra_instruments_presence_tensor_dataset,
                                 total_chunk_counter, too_many_instruments_frame, impossible_transposition):

        ############################################################
        # Transpose pianorolls
        this_pr_piano = shift_pr_along_pitch_axis(pianorolls_pair["Piano"], transposition_semi_tone)
        this_onsets_piano = shift_pr_along_pitch_axis(onsets_pair["Piano"], transposition_semi_tone)

        this_pr_orchestra = {}
        this_onsets_orchestra = {}
        for instrument_name in pianorolls_pair["Orchestra"].keys():
            # Pr
            pr = pianorolls_pair["Orchestra"][instrument_name]
            shifted_pr = shift_pr_along_pitch_axis(pr, transposition_semi_tone)
            this_pr_orchestra[instrument_name] = shifted_pr
            # Onsets
            onsets = onsets_pair["Orchestra"][instrument_name]
            shifted_onsets = shift_pr_along_pitch_axis(onsets, transposition_semi_tone)
            this_onsets_orchestra[instrument_name] = shifted_onsets
        ############################################################

        if minimum_transposition_allowed is None:
            if transposition_semi_tone != 0:
                raise Exception("Possible transpositions should be computed on non transposed pianorolls")
            # We have to construct the possible transpose
            build_allowed_transposition_flag = True
            minimum_transposition_allowed = []
            maximum_transposition_allowed = []
        else:
            build_allowed_transposition_flag = False

        for chunk_index in range(len(chunks_piano_indices)):
            this_chunk_piano_indices = chunks_piano_indices[chunk_index]
            this_chunk_orchestra_indices = chunks_orchestra_indices[chunk_index]
            avoid_this_chunk = False
            total_chunk_counter += 1

            ############################################################
            if build_allowed_transposition_flag:
                min_transposition = -self.max_transposition
                max_transposition = self.max_transposition

                # Observe tessitura for each instrument for this chunk. Use non transposed pr of course
                this_min_transposition, this_max_transposition = \
                    self.get_allowed_transpositions_from_pr(this_pr_piano,
                                                            this_chunk_piano_indices,
                                                            "Piano")

                min_transposition = max(this_min_transposition, min_transposition)
                max_transposition = min(this_max_transposition, max_transposition)

                # Use reference tessitura or compute tessitura directly on the files ?
                for instrument_name, pr in this_pr_orchestra.items():
                    this_min_transposition, this_max_transposition = \
                        self.get_allowed_transpositions_from_pr(pr,
                                                                this_chunk_orchestra_indices,
                                                                instrument_name)
                    if this_min_transposition is not None:  # If instrument not in this chunk, None was returned
                        min_transposition = max(this_min_transposition, min_transposition)
                        max_transposition = min(this_max_transposition, max_transposition)

                this_minimum_transposition_allowed = min(0, min_transposition)
                this_maximum_transposition_allowed = max(0, max_transposition)
                minimum_transposition_allowed.append(this_minimum_transposition_allowed)
                maximum_transposition_allowed.append(this_maximum_transposition_allowed)
            else:
                this_minimum_transposition_allowed = minimum_transposition_allowed[chunk_index]
                this_maximum_transposition_allowed = maximum_transposition_allowed[chunk_index]
            ############################################################

            #  Test if the transposition is possible
            if (this_minimum_transposition_allowed > transposition_semi_tone) \
                    or (this_maximum_transposition_allowed < transposition_semi_tone):
                impossible_transposition += 1
                continue

            ############################################################
            local_piano_tensor = []
            local_orchestra_tensor = []
            local_orchestra_instruments_presence_tensor = []
            previous_notes_orchestra = None
            for frame_piano, frame_orchestra in zip(this_chunk_piano_indices, this_chunk_orchestra_indices):
                # Piano encoded vector
                if frame_orchestra in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                    #  Padding vectors at beginning or end
                    piano_t_encoded = self.precomputed_vectors_piano[frame_piano].clone().detach()
                    orchestra_t_encoded = self.precomputed_vectors_orchestra[frame_orchestra].clone().detach()
                    orchestra_instruments_presence_t_encoded = \
                        self.precomputed_vectors_orchestra_instruments_presence[PAD_SYMBOL].clone().detach()
                else:
                    piano_t_encoded = self.pianoroll_to_piano_tensor(
                        pr=this_pr_piano,
                        onsets=this_onsets_piano,
                        frame_index=frame_piano
                    )
                    orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = \
                        self.pianoroll_to_orchestral_tensor(
                            pr=this_pr_orchestra,
                            onsets=this_onsets_orchestra,
                            previous_notes=previous_notes_orchestra,
                            frame_index=frame_orchestra
                        )

                if orchestra_t_encoded is None:
                    avoid_this_chunk = True
                    break

                local_piano_tensor.append(piano_t_encoded)
                local_orchestra_tensor.append(orchestra_t_encoded)
                local_orchestra_instruments_presence_tensor.append(orchestra_instruments_presence_t_encoded)
            ############################################################

            if avoid_this_chunk:
                too_many_instruments_frame += 1
                continue

            assert len(local_piano_tensor) == self.sequence_size
            assert len(local_orchestra_tensor) == self.sequence_size

            local_piano_tensor = torch.stack(local_piano_tensor)
            local_orchestra_tensor = torch.stack(local_orchestra_tensor)
            local_orchestra_instruments_presence_tensor = torch.stack(local_orchestra_instruments_presence_tensor)

            piano_tensor_dataset.append(
                local_piano_tensor[None, :, :].int())
            orchestra_tensor_dataset.append(
                local_orchestra_tensor[None, :, :].int())
            orchestra_instruments_presence_tensor_dataset.append(
                local_orchestra_instruments_presence_tensor[None, :, :].int())

        return minimum_transposition_allowed, maximum_transposition_allowed, \
               piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
               total_chunk_counter, too_many_instruments_frame, impossible_transposition

    def pianoroll_to_piano_tensor(self, pr, onsets, frame_index):
        piano_encoded = np.zeros((self.number_pitch_piano))
        #  Write one-hot
        for midi_pitch, index in self.midi_pitch2index_piano.items():
            this_velocity = pr[frame_index, midi_pitch]
            if (this_velocity != 0) and (onsets[frame_index, midi_pitch] == 0):
                piano_encoded[index] = self.value2oneHot_perPianoToken[index][SLUR_SYMBOL]
            elif this_velocity == 0:
                piano_encoded[index] = self.value2oneHot_perPianoToken[index][REST_SYMBOL]
            else:
                piano_encoded[index] = self.value2oneHot_perPianoToken[index][this_velocity]
        piano_tensor = torch.from_numpy(piano_encoded).long()
        return piano_tensor

    def pianoroll_to_orchestral_tensor(self, pr, onsets, previous_notes, frame_index):
        """
        previous_notes = {'instrument_name': {'note': 'index'}}
        maintain a list of notes and their index, regardless of slurs

        :param pr:
        :param onsets:
        :param previous_notes:
        :param frame_index:
        :return:
        """

        orchestra_encoded = np.zeros((self.number_instruments)) - 1
        orchestra_instruments_presence = np.zeros((len(self.instrument_presence_index2name)))

        current_notes = {}

        for instrument_name, indices_instruments in self.instrument2index.items():

            current_notes[instrument_name] = {}

            # Avoid messing aroud with indices
            this_instrument_midi2index = self.midi_pitch2index[indices_instruments[0]]

            if instrument_name not in pr.keys():
                for index in indices_instruments:
                    if orchestra_encoded[index] == -1:
                        orchestra_encoded[index] = this_instrument_midi2index[REST_SYMBOL]
                continue

            # Get list of note at frame_index
            if previous_notes is None:
                notes_onsets = [e for e in list(np.where(pr[instrument_name][frame_index])[0])]
                notes_slurs = []
            else:
                notes_onsets = [e for e in list(np.where(onsets[instrument_name][frame_index])[0])]
                notes_slurs = [e for e in list(np.where(pr[instrument_name][frame_index])[0]) if e not in notes_onsets]

            # Sort note from lowest to highest
            notes_onsets = sorted(notes_onsets)
            notes_slurs = sorted(notes_slurs)

            # Instrument_presence_vector
            instrument_presence_index = self.instrument_presence_name2index[instrument_name]
            if (len(notes_onsets) == 0) and (len(notes_slurs) == 0):
                orchestra_instruments_presence[instrument_presence_index] = self.instruments_presence2index[NO_SYMBOL]
            else:
                orchestra_instruments_presence[instrument_presence_index] = self.instruments_presence2index[YES_SYMBOL]

            #  First write Slurs at same location than slured not
            for note in notes_slurs:
                # OOR ?
                if note in this_instrument_midi2index.keys():
                    encoded_note = this_instrument_midi2index[note]
                else:
                    # skip
                    break

                writen = False
                #  Search in previous frame
                for previous_note, previous_index in previous_notes[instrument_name].items():
                    if previous_note == encoded_note:
                        orchestra_encoded[previous_index] = this_instrument_midi2index[SLUR_SYMBOL]
                        writen = True
                        current_notes[instrument_name][encoded_note] = previous_index
                        break

                if not writen:
                    #  Can happen if its the first frame or onset is not up to date anymore after automatic alignement
                    # (due to skipped frames)
                    for index in indices_instruments:
                        if orchestra_encoded[index] == -1:
                            orchestra_encoded[index] = encoded_note
                            current_notes[instrument_name][encoded_note] = index
                            writen = True
                            break

            #  Write onsets notes at other locations
            for note in notes_onsets:
                #  Find first free slot
                if note in this_instrument_midi2index.keys():
                    encoded_note = this_instrument_midi2index[note]
                else:
                    # skip
                    print(f'OOR: {instrument_name} - {note}')
                    break
                for index in indices_instruments:
                    if orchestra_encoded[index] == -1:
                        orchestra_encoded[index] = encoded_note
                        current_notes[instrument_name][encoded_note] = index
                        break

            #  Fill with silences
            for index in indices_instruments:
                if orchestra_encoded[index] == -1:
                    orchestra_encoded[index] = this_instrument_midi2index[REST_SYMBOL]

        orchestra_tensor = torch.from_numpy(orchestra_encoded).long()
        orchestra_instruments_presence_tensor = torch.from_numpy(orchestra_instruments_presence).long()

        return orchestra_tensor, current_notes, orchestra_instruments_presence_tensor

    def extract_score_tensor_with_padding(self, tensor_score):
        return None

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        return None

    def empty_score_tensor(self, score_length):

        return None

    def random_score_tensor(self, score_length):
        return None

    def piano_tensor_to_score(self, tensor_score, format, durations, subdivision):

        piano_matrix = tensor_score.numpy()
        length = len(piano_matrix)

        if subdivision is None:
            subdivision = self.subdivision

        if durations is None:
            durations = np.ones((length)) * subdivision
        assert length == len(durations)

        # Instrument
        this_part = music21.stream.Part(id='Piano')
        music21_instrument = music21.instrument.fromString('Piano')
        this_part.insert(music21_instrument)

        # Browse pitch dimension first, to deal with sustained notes
        for piano_index, pitch in self.index2midi_pitch_piano.items():
            offset = 0
            duration = 0
            current_offset = 0
            velocity = None
            # f = None
            for frame_index in range(length):
                current_velocity = self.oneHot2value_perPianoToken[piano_index][piano_matrix[frame_index, piano_index]]
                current_duration = durations[frame_index]

                # Write note if current note is not slured
                if current_velocity != SLUR_SYMBOL:
                    #  Write previous frame if it was not a silence
                    if velocity is not None:
                        if velocity not in [REST_SYMBOL, START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                            f = music21.note.Note(pitch)
                            f.volume.velocity = unquantize_velocity(velocity, self.velocity_quantization)
                            f.quarterLength = duration / subdivision
                            this_part.insert((offset / subdivision), f)
                        # Reinitialise (note that we don't need to write silences, they are handled by the offset)
                        # else:
                        #     f = music21.note.Rest()
                        #     f.quarterLength = duration / subdivision
                        #     this_part.insert((offset / subdivision), f)
                    duration = current_duration
                    offset = current_offset
                    velocity = current_velocity
                elif current_velocity == SLUR_SYMBOL:
                    duration += current_duration

                current_offset += current_duration

            # Don't forget the last note
            # if velocity not in [REST_SYMBOL]:
            if velocity not in [REST_SYMBOL]:
                f = music21.note.Note(pitch)
                f.volume.velocity = unquantize_velocity(velocity, self.velocity_quantization)
                f.quarterLength = duration / subdivision
                this_part.insert((offset / subdivision), f)

        if format == 'xml':
            this_part = this_part.chordify()

        return this_part

    def orchestra_tensor_to_dict(self, tensor_score, durations, subdivision):
        # (batch, num_parts, notes_encoding)
        orchestra_matrix = tensor_score.numpy()
        length = len(orchestra_matrix)

        if subdivision is None:
            subdivision = self.subdivision

        if durations is None:
            durations = np.ones((length)) * subdivision
        else:
            assert length == len(durations), "Rhythm vector must be the same length as tensor[0]"

        total_duration_ql = sum(durations) / subdivision

        # First store every in a dict {instrus : [time [notes]]}
        score_dict = {}
        for instrument_index in range(self.number_instruments):
            # Get instrument name
            instrument_name = self.index2instrument[instrument_index]
            if instrument_name not in score_dict:
                score_dict[instrument_name] = []

            # First store every in a dict {instrus : [time [notes]]}
            score_list = []
            offset = 0

            for frame_index, duration in enumerate(durations):
                symbol = self.index2midi_pitch[instrument_index][orchestra_matrix[frame_index, instrument_index]]
                if symbol not in [START_SYMBOL, END_SYMBOL, REST_SYMBOL, PAD_SYMBOL]:
                    if symbol == SLUR_SYMBOL:
                        if len(score_list) == 0:
                            print(f'Slur symbol placed after nothing in {instrument_name}')
                            continue
                        else:
                            (this_pitch, this_offset, this_duration) = score_list.pop(-1)
                        new_elem = (this_pitch, this_offset, this_duration + duration)
                        score_list.append(new_elem)
                    else:
                        new_elem = (symbol, offset, duration)
                        score_list.append(new_elem)
                offset += duration

            score_dict[instrument_name] += score_list

        return score_dict, total_duration_ql

    def orchestra_tensor_to_score(self, tensor_score, format, durations, subdivision):
        """

        :param durations:
        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :return:
        """

        score_dict, total_duration_ql = self.orchestra_tensor_to_dict(tensor_score, durations, subdivision)

        #  Batch is used as time in the score
        stream = music21.stream.Stream()
        streams = {}

        for instrument_name, elems in score_dict.items():
            this_part = music21.stream.Part(id=instrument_name)
            #  re is for removing underscores in instrument names which raise errors in music21
            if instrument_name == "Cymbal":
                music21_instrument = music21.instrument.Cymbals()
            elif instrument_name == "Woodwind":
                music21_instrument = music21.instrument.fromString("Clarinet")
            elif instrument_name == "String":
                music21_instrument = music21.instrument.fromString("Violoncello")
            elif instrument_name == "Brass":
                music21_instrument = music21.instrument.fromString("Horn")
            else:
                music21_instrument = music21.instrument.fromString(re.sub('_', ' ', instrument_name))
            this_part.insert(0, music21_instrument)

            if elems == []:
                f = music21.note.Rest()
                f.quarterLength = total_duration_ql
                this_part.insert(0, f)
            else:
                #  Sort by offset time (not sure it's very useful, more for debugging purposes)
                elems = sorted(elems, key=lambda e: e[1])
                for elem in elems:
                    pitch, offset, duration = elem
                    try:
                        f = music21.note.Note(pitch)
                    except:
                        print('yo')
                    f.volume.velocity = 60.
                    f.quarterLength = duration / subdivision
                    this_part.insert((offset / subdivision), f)

            if format == 'xml':
                this_part = this_part.chordify()
            this_part.atSoundingPitch = self.transpose_to_sounding_pitch
            stream.append(this_part)
            streams[instrument_name] = this_part

        return stream, streams, score_dict

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano, writing_dir,
                        filepath, subdivision, only_orchestra):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{self.dump_folder}/arrangement"

        if len(piano_pianoroll.size()) == 2:
            piano_flat = piano_pianoroll
            orchestra_flat = orchestra_pianoroll
        else:
            # Add padding vectors between each example
            batch_size, time_length, num_features = piano_pianoroll.size()
            piano_with_padding_between_batch = torch.zeros(batch_size, time_length + 1, num_features)
            piano_with_padding_between_batch[:, :time_length] = piano_pianoroll
            piano_with_padding_between_batch[:, time_length] = self.precomputed_vectors_piano[REST_SYMBOL]
            piano_flat = piano_with_padding_between_batch.view(-1, self.number_pitch_piano)
            #
            batch_size, time_length, num_features = orchestra_pianoroll.size()
            orchestra_with_padding_between_batch = torch.zeros(batch_size, time_length + 1, num_features)
            orchestra_with_padding_between_batch[:, :time_length] = orchestra_pianoroll
            orchestra_with_padding_between_batch[:, time_length] = self.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_flat = orchestra_with_padding_between_batch.view(-1, self.number_instruments)

        piano_part = self.piano_tensor_to_score(piano_flat, durations_piano,
                                                subdivision=subdivision)
        orchestra_stream, _, _ = self.orchestra_tensor_to_score(orchestra_flat, durations_piano,
                                                                durations=None,
                                                                subdivision=subdivision)

        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_orchestra.mid", fmt='midi')
        if not only_orchestra:
            piano_part.write(fp=f"{writing_dir}/{filepath}_piano.mid", fmt='midi')
            # Both in the same score
            orchestra_stream.append(piano_part)
            orchestra_stream.write(fp=f"{writing_dir}/{filepath}_both.mid", fmt='midi')

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):
        # Get pianorolls
        score_piano = music21.converter.parse(filepath)

        if subdivision is None:
            subdivision = self.subdivision
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(score_piano,
                                                              subdivision,
                                                              simplify_instrumentation=None,
                                                              instrument_grouping=self.instrument_grouping,
                                                              transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
                                                              integrate_discretization=self.integrate_discretization,
                                                              binarize=False)
        return self.pianoroll_to_formated_tensor(pianoroll_piano, onsets_piano, batch_size, context_length,
                                                 banned_instruments, unknown_instruments)

    def pianoroll_to_formated_tensor(self, pianoroll_piano, onsets_piano, batch_size, context_length,
                                     banned_instruments, unknown_instruments):
        quantized_pianoroll_piano = quantize_velocity_pianoroll_frame(pianoroll_piano["Piano"],
                                                                      self.velocity_quantization)

        rhythm_piano = new_events(pianoroll_piano, onsets_piano)
        onsets_piano = onsets_piano["Piano"]
        piano_tensor = []
        for frame_index in rhythm_piano:
            piano_t_encoded = self.pianoroll_to_piano_tensor(
                quantized_pianoroll_piano,
                onsets_piano,
                frame_index)
            piano_tensor.append(piano_t_encoded)

        # Prepend rests frames at the beginning and end of the piano score
        piano_tensor = [self.precomputed_vectors_piano[PAD_SYMBOL]] * (context_length - 1) + \
                       [self.precomputed_vectors_piano[START_SYMBOL]] + \
                       piano_tensor + \
                       [self.precomputed_vectors_piano[END_SYMBOL]] + \
                       [self.precomputed_vectors_piano[PAD_SYMBOL]] * (context_length - 1)

        piano_init = torch.stack(piano_tensor)

        # Orchestra
        num_frames = piano_init.shape[0]  #  Here batch size is time dimensions (each batch index is a piano event)
        orchestra_silences, orchestra_unknown, instruments_presence, orchestra_init = \
            self.init_orchestra(num_frames, context_length, banned_instruments, unknown_instruments)

        # Repeat along batch dimension to generate several orchestation of the same piano score
        piano_init = piano_init.unsqueeze(0).repeat(batch_size, 1, 1)
        orchestra_init = orchestra_init.unsqueeze(0).repeat(batch_size, 1, 1)
        instruments_presence_init = instruments_presence.unsqueeze(0).repeat(batch_size, 1, 1)
        piano_write = piano_init

        return piano_init.long(), piano_write.long(), rhythm_piano, \
               orchestra_init.long(), \
               instruments_presence_init.long(), orchestra_silences, orchestra_unknown

    def init_orchestra(self, num_frames, context_length, banned_instruments, unknown_instruments):
        # Set orchestra constraints in the form of banned instruments
        orchestra_silences = []
        orchestra_unknown = []
        instruments_presence = torch.zeros(num_frames, self.instrument_presence_dim)
        orchestra_init = torch.zeros(num_frames, self.number_instruments)
        for instrument_name, instrument_indices in self.instrument2index.items():
            if instrument_name in banned_instruments:
                instrument_presence_value = self.instruments_presence2index[NO_SYMBOL]
            else:
                instrument_presence_value = self.instruments_presence2index[YES_SYMBOL]
            instruments_presence[:, self.instrument_presence_name2index[instrument_name]] = instrument_presence_value
            for instrument_index in instrument_indices:
                if instrument_name in banned_instruments:
                    # -1 is a silence
                    orchestra_silences.append(1)
                    orchestra_init[:, instrument_index] = self.midi_pitch2index[instrument_index][REST_SYMBOL]
                elif instrument_name in unknown_instruments:
                    # Note that an instrument can't be both banned and unknown
                    orchestra_unknown.append(1)
                else:
                    orchestra_silences.append(0)
                    orchestra_unknown.append(0)
                    #  Initialise with last
                    orchestra_init[:, instrument_index] = len(self.midi_pitch2index[instrument_index])

        # Start and end symbol at the beginning and end
        orchestra_init[:context_length - 1] = self.precomputed_vectors_orchestra[PAD_SYMBOL]
        orchestra_init[context_length - 1] = self.precomputed_vectors_orchestra[START_SYMBOL]
        orchestra_init[-context_length] = self.precomputed_vectors_orchestra[END_SYMBOL]
        orchestra_init[-context_length + 1:] = self.precomputed_vectors_orchestra[PAD_SYMBOL]
        return orchestra_silences, orchestra_unknown, instruments_presence, orchestra_init


if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

    database_path = f'{os.path.expanduser("~")}/Data/databases'

    # parameters
    sequence_size = 5
    max_transposition = 12
    velocity_quantization = 2
    subdivision = 16
    integrate_discretization = True

    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path=f'{database_path}/Orchestration/arrangement',
        subsets=[
            'liszt_classical_archives',
        ],
        num_elements=None
    )

    # corpus_it_gen_instru_range = OrchestraIteratorGenerator(
    #     folder_path=f"{config['database_path']}/Orchestration/orchestral",
    #     subsets=[
    #         "kunstderfuge"
    #     ],
    #     process_file=True,
    # )
    corpus_it_gen_instru_range = None

    dataset = ArrangementDataset(corpus_it_gen=corpus_it_gen,
                                 corpus_it_gen_instru_range=corpus_it_gen_instru_range,
                                 name='shit',
                                 subdivision=subdivision,
                                 sequence_size=sequence_size,
                                 velocity_quantization=velocity_quantization,
                                 max_transposition=max_transposition,
                                 integrate_discretization=integrate_discretization,
                                 alignement_type='complete',
                                 transpose_to_sounding_pitch=True,
                                 compute_statistics_flag=None)

    dataset.load_index_dicts()

    dump_folder = dataset.dump_folder
    writing_dir = f'{dump_folder}/arrangement/reconstruction_midi'
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)

    num_frames = 500

    for arr_pair in dataset.iterator_gen():
        ######################################################################
        # Piano
        # Tensor
        arr_id = arr_pair['name']

        pianoroll_piano, onsets_piano, _ = \
            score_to_pianoroll(arr_pair['Piano'],
                               subdivision,
                               simplify_instrumentation=None,
                               instrument_grouping=dataset.instrument_grouping,
                               transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
                               integrate_discretization=dataset.integrate_discretization,
                               binarize=False)
        # Quantize piano
        pr_piano = quantize_velocity_pianoroll_frame(pianoroll_piano["Piano"],
                                                     velocity_quantization)

        events_piano = new_events(pianoroll_piano, onsets_piano)
        events_piano = events_piano[:num_frames]
        onsets_piano = onsets_piano["Piano"]
        piano_tensor = []
        for frame_index in events_piano:
            piano_frame = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano,
                onsets=onsets_piano,
                frame_index=frame_index)

            piano_tensor.append(piano_frame)

        piano_tensor = torch.stack(piano_tensor)

        # Reconstruct
        piano_cpu = piano_tensor.cpu()
        duration_piano = list(np.asarray(events_piano)[1:] - np.asarray(events_piano)[:-1]) + [subdivision]

        piano_part = dataset.piano_tensor_to_score(piano_cpu, duration_piano, subdivision=subdivision)
        # piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.xml", fmt='musicxml')
        piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.mid", fmt='midi')

        ######################################################################
        #  Orchestra
        pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(
            score=arr_pair["Orchestra"],
            subdivision=subdivision,
            simplify_instrumentation=dataset.simplify_instrumentation,
            instrument_grouping=dataset.instrument_grouping,
            transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
            integrate_discretization=dataset.integrate_discretization,
            binarize=False)

        events_orchestra = new_events(pianoroll_orchestra, onsets_orchestra)
        events_orchestra = events_orchestra[:num_frames]
        orchestra_tensor = []
        previous_notes_orchestra = None
        for frame_counter, frame_index in enumerate(events_orchestra):
            orchestra_t_encoded, previous_notes_orchestra, _ = dataset.pianoroll_to_orchestral_tensor(
                pr=pianoroll_orchestra,
                onsets=onsets_orchestra,
                previous_notes=previous_notes_orchestra,
                frame_index=frame_index)
            if orchestra_t_encoded is None:
                orchestra_t_encoded = dataset.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_tensor.append(orchestra_t_encoded)

        orchestra_tensor = torch.stack(orchestra_tensor)

        # Reconstruct
        orchestra_cpu = orchestra_tensor.cpu()
        duration_orchestra = list(np.asarray(events_orchestra)[1:] - np.asarray(events_orchestra)[:-1]) + [
            subdivision]
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, duration_orchestra,
                                                           subdivision=subdivision)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_orchestra.mid", fmt='midi')

        ######################################################################
        # Original
        try:
            arr_pair["Orchestra"].write(fp=f"{writing_dir}/{arr_id}_original.mid", fmt='midi')
            arr_pair["Piano"].write(fp=f"{writing_dir}/{arr_id}_original_piano.mid", fmt='midi')
        except:
            print("Can't write original")

        ######################################################################
        # Aligned version
        corresponding_frames, this_scores = dataset.align_score(pianoroll_piano,
                                                                onsets_piano,
                                                                pianoroll_orchestra,
                                                                onsets_orchestra)
        corresponding_frames = corresponding_frames[:num_frames]

        piano_frames = [e[0][0] for e in corresponding_frames]
        orchestra_frames = [e[1][0] for e in corresponding_frames]

        # Transpose pianorolls
        transposition_semi_tone = 1
        pr_piano_transp = shift_pr_along_pitch_axis(pr_piano, transposition_semi_tone)
        onsets_piano_transp = shift_pr_along_pitch_axis(onsets_piano, transposition_semi_tone)

        pr_orchestra_transp = {}
        onsets_orchestra_transp = {}
        for instrument_name in pianoroll_orchestra.keys():
            # Pr
            pr = pianoroll_orchestra[instrument_name]
            shifted_pr = shift_pr_along_pitch_axis(pr, transposition_semi_tone)
            pr_orchestra_transp[instrument_name] = shifted_pr
            # Onsets
            onsets = onsets_orchestra[instrument_name]
            shifted_onsets = shift_pr_along_pitch_axis(onsets, transposition_semi_tone)
            onsets_orchestra_transp[instrument_name] = shifted_onsets

        piano_tensor_event = []
        orchestra_tensor_event = []
        previous_frame_index = None
        previous_notes_orchestra = None
        for frame_counter, (frame_piano, frame_orchestra) in enumerate(zip(piano_frames, orchestra_frames)):

            #  IMPORTANT:
            #  Compute orchestra first to know if the frame has to be skipped or not
            #  (typically if too many instruments are played in one section)

            #######
            # Orchestra
            orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = \
                dataset.pianoroll_to_orchestral_tensor(
                    pr=pr_orchestra_transp,
                    onsets=onsets_orchestra_transp,
                    previous_notes=previous_notes_orchestra,
                    frame_index=frame_orchestra
                )

            if orchestra_t_encoded is None:
                avoid_this_chunk = True
                continue
            orchestra_tensor_event.append(orchestra_t_encoded)

            #######
            # Piano
            piano_t = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano_transp,
                onsets=onsets_piano_transp,
                frame_index=frame_piano)
            piano_tensor_event.append(piano_t)

        piano_tensor_event = torch.stack(piano_tensor_event)
        orchestra_tensor_event = torch.stack(orchestra_tensor_event)
        # Reconstruct
        orchestra_cpu = orchestra_tensor_event.cpu()
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, durations=None, subdivision=subdivision)
        piano_cpu = piano_tensor_event.cpu()
        piano_part = dataset.piano_tensor_to_score(piano_cpu, durations=None, subdivision=subdivision)
        orchestra_part.append(piano_part)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_both_aligned.mid", fmt='midi')
