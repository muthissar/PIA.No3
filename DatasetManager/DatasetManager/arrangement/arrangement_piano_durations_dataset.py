# TODO Use future piano information ?
import json
import os
import re
import shutil

import torch
import matplotlib as mpl
from torch.utils.data import TensorDataset

mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import music21
import numpy as np

from DatasetManager.arrangement.instrumentation import get_instrumentation
from DatasetManager.helpers import PAD_SYMBOL, REST_SYMBOL, SLUR_SYMBOL, DURATION_SYMBOL
from DatasetManager.music_dataset import MusicDataset
import DatasetManager.arrangement.nw_align as nw_align

from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll, quantize_and_filter_music21_element, \
    quantize_velocity_pianoroll_frame, unquantize_velocity


class ArrangementPianoDurationDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 subdivision=2,
                 sequence_size=2,
                 velocity_quantization=8,
                 max_transposition=3,
                 transpose_to_sounding_pitch=True,
                 reference_tessitura_path='/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/reference_tessitura.json',
                 simplify_instrumentation_path='/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/simplify_instrumentation.json',
                 cache_dir=None,
                 compute_statistics_flag=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super(ArrangementPianoDurationDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.subdivision = subdivision  # We use only on beats notes so far
        self.sequence_size = sequence_size
        self.velocity_quantization = velocity_quantization
        self.max_transposition = max_transposition
        self.transpose_to_sounding_pitch = transpose_to_sounding_pitch
        # Perhaps we should set it to zero since it's quite fucked up for now
        self.max_duration = 20


        # Reference tessitura for each instrument
        with open(reference_tessitura_path, 'r') as ff:
            tessitura = json.load(ff)
        self.reference_tessitura = {k: (music21.note.Note(v[0]), music21.note.Note(v[1])) for k, v in tessitura.items()}
        self.observed_tessitura = {}

        # Maps parts name found in mxml files to standard names
        with open(simplify_instrumentation_path, 'r') as ff:
            self.simplify_instrumentation = json.load(ff)

        #  Instrumentation used for learning
        self.instrumentation = get_instrumentation()

        # Mapping between instruments and indices
        self.index2instrument = {}
        self.instrument2index = {}
        self.index2midi_pitch = {}
        self.midi_pitch2index = {}
        #  Piano
        self.midi_pitch_velocity2index_piano = {}
        self.midi_pitch_duration2index_piano = {}
        self.index2midi_pitch_velocity_piano = {}
        self.index2midi_pitch_duration_piano = {}
        self.value2oneHot_perPianoToken = {}
        self.oneHot2value_perPianoToken = {}
        # Dimensions
        self.number_instruments = None
        self.number_pitch_piano = None
        # Padding vectors (built in compute_index_dicts)
        self.piano_padding_vector = None
        self.orchestra_padding_vector = None

        # Compute statistics slows down the construction of the dataset
        self.compute_statistics_flag = compute_statistics_flag
        return

    def __repr__(self):
        return f'ArrangementFrameDataset(' \
            f'{self.name},' \
            f'{self.subdivision})'

    def iterator_gen(self):
        return (self.sort_arrangement_pairs(arrangement_pair)
                for arrangement_pair in self.corpus_it_gen())

    def compute_index_dicts(self):
        #  Mapping midi_pitch to token for each instrument
        set_midiPitch_per_instrument = {'Piano': set()}
        # First pass over the database to create the mapping pitch <-> index for each instrument
        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):
            # Compute pianoroll representations of score (more efficient than manipulating the music21 streams)
            pianoroll_piano, onsets_piano, _ = score_to_pianoroll(arr_pair['Piano'], self.subdivision,
                                                                  self.simplify_instrumentation,
                                                                  self.transpose_to_sounding_pitch)
            pitch_set_this_track = set(np.where(np.sum(pianoroll_piano['Piano'], axis=0) > 0)[0])
            set_midiPitch_per_instrument['Piano'] = set_midiPitch_per_instrument['Piano'].union(pitch_set_this_track)

            pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(arr_pair['Orchestra'], self.subdivision,
                                                                          self.simplify_instrumentation,
                                                                          self.transpose_to_sounding_pitch)
            for instrument_name in pianoroll_orchestra:
                if instrument_name not in set_midiPitch_per_instrument.keys():
                    set_midiPitch_per_instrument[instrument_name] = set()
                pitch_set_this_track = set(np.where(np.sum(pianoroll_orchestra[instrument_name], axis=0) > 0)[0])
                set_midiPitch_per_instrument[instrument_name] = set_midiPitch_per_instrument[instrument_name].union(
                    pitch_set_this_track)

        # Save this in a file
        if self.compute_statistics_flag:
            with open(f"{self.compute_statistics_flag}/pc_frequency_per_instrument", "w") as ff:
                for instrument_name, set_pitch_class in set_midiPitch_per_instrument.items():
                    ff.write(f"# {instrument_name}: \n")
                    for pc in set_pitch_class:
                        ff.write(f"   {pc}\n")

        # Local dicts used temporarily
        midi_pitch2index_per_instrument = {}
        index2midi_pitch_per_instrument = {}
        for instrument_name, set_midiPitch in set_midiPitch_per_instrument.items():
            min_pitch = min(set_midiPitch)
            max_pitch = max(set_midiPitch)
            self.observed_tessitura[instrument_name] = {
                "min": min_pitch,
                "max": max_pitch
            }
            if instrument_name == "Piano":
                continue
            # Use range to avoid gaps in instruments tessitura (needed since we use
            # pitch transpositions as data augmentations
            list_midiPitch = sorted(list(range(min_pitch, max_pitch + 1)))
            midi_pitch2index_per_instrument[instrument_name] = {}
            index2midi_pitch_per_instrument[instrument_name] = {}
            for index, midi_pitch in enumerate(list_midiPitch):
                midi_pitch2index_per_instrument[instrument_name][midi_pitch] = index
                index2midi_pitch_per_instrument[instrument_name][index] = midi_pitch
            # Silence
            index += 1
            midi_pitch2index_per_instrument[instrument_name][REST_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = REST_SYMBOL
            #  Slur
            index += 1
            midi_pitch2index_per_instrument[instrument_name][SLUR_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = SLUR_SYMBOL
            # Pad
            index += 1
            midi_pitch2index_per_instrument[instrument_name][PAD_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = PAD_SYMBOL

        # Mapping instruments <-> indices
        index_counter = 0
        for instrument_name, number_instruments in self.instrumentation.items():
            #  Check if instrument appears in the dataset
            if instrument_name not in midi_pitch2index_per_instrument.keys():
                continue
            self.instrument2index[instrument_name] = list(range(index_counter, index_counter + number_instruments))
            for ind in range(index_counter, index_counter + number_instruments):
                self.index2instrument[ind] = instrument_name
            index_counter += number_instruments

        # Mapping pitch <-> index per voice (that's the one we'll use, easier to manipulate when training)
        for instrument_name, instrument_indices in self.instrument2index.items():
            for instrument_index in instrument_indices:
                self.midi_pitch2index[instrument_index] = midi_pitch2index_per_instrument[instrument_name]
                self.index2midi_pitch[instrument_index] = index2midi_pitch_per_instrument[instrument_name]

        # Piano
        min_pitch_piano = min(set_midiPitch_per_instrument["Piano"])
        max_pitch_piano = max(set_midiPitch_per_instrument["Piano"])
        #  Use range to avoid "gaps" in the piano tessitura
        list_midiPitch = sorted(list(range(min_pitch_piano, max_pitch_piano + 1)))
        for index, midi_pitch in enumerate(list_midiPitch):
            #  Velocity (even indices)
            self.midi_pitch_velocity2index_piano[midi_pitch] = 2*index
            self.index2midi_pitch_velocity_piano[2*index] = midi_pitch
            # Duration (odd indices)
            self.midi_pitch_duration2index_piano[midi_pitch] = 2*index+1
            self.index2midi_pitch_duration_piano[2*index+1] = midi_pitch

        # One hot encoding for velocitites
        dict_for_velocity2oneHot = {}
        dict_for_oneHot2velocity = {}
        for index, velocity in enumerate(range(self.velocity_quantization)):
            dict_for_velocity2oneHot[velocity] = index
            dict_for_oneHot2velocity[index] = velocity
        #  Slur
        index += 1
        dict_for_velocity2oneHot[SLUR_SYMBOL] = index
        dict_for_oneHot2velocity[index] = SLUR_SYMBOL
        # One hot encoding for durations (no SLUR for durations)
        dict_for_duration2oneHot = {}
        dict_for_oneHot2duration = {}
        for index, duration in enumerate(range(self.max_duration)):
            dict_for_duration2oneHot[duration] = index
            dict_for_oneHot2duration[index] = duration

        for token_index, symbol_pitch_or_duration in self.index2midi_pitch_piano.items():
            if symbol_pitch_or_duration == DURATION_SYMBOL:
                self.value2oneHot_perPianoToken[token_index] = {i: i for i in range(1, self.max_duration)}
                self.oneHot2value_perPianoToken[token_index] = {i: i for i in range(1, self.max_duration)}
                # Padding is stored as 0 duration
                self.value2oneHot_perPianoToken[token_index][PAD_SYMBOL] = 0
                self.oneHot2value_perPianoToken[token_index][0] = PAD_SYMBOL
            else:
                self.value2oneHot_perPianoToken[token_index] = dict.copy(dict_for_velocity2oneHot)
                self.oneHot2value_perPianoToken[token_index] = dict.copy(dict_for_oneHot2velocity)

        self.number_instruments = len(self.midi_pitch2index)
        self.number_pitch_piano = len(self.midi_pitch2index_piano)

        # Need those two for padding sequences
        self.piano_padding_vector = torch.from_numpy(np.asarray([0] * self.number_pitch_piano)).long()
        # todo  REMARQUE : In padding piano vector, we mark them with the two lowest notes
        #  being played (just for debugging, use silences when training)
        self.piano_padding_vector[0:2] = self.velocity_quantization - 1
        orchestra_padding_vector = [mapping[PAD_SYMBOL] for instru_ind, mapping in self.midi_pitch2index.items()]
        self.orchestra_padding_vector = torch.from_numpy(np.asarray(orchestra_padding_vector)).long()
        return

    def make_tensor_dataset(self, frame_orchestra=None):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        self.compute_index_dicts()

        # one_tick = 1 / self.subdivision
        piano_tensor_dataset = []
        orchestra_tensor_dataset = []
        # metadata_tensor_dataset = []

        total_chunk_counter = 0
        missed_chunk_counter = 0
        total_chunk_counter_augmented = 0
        missed_chunk_counter_augmented = 0

        # Variables for statistics
        if self.compute_statistics_flag:
            scores = []
            num_frames_with_different_pitch_class = 0
            total_frames_counter = 0
            open(f"{self.compute_statistics_flag}/different_set_pc.txt", 'w').close()

        # Iterate over files
        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

            ############################################################
            # Precompute transpositions
            pianoroll_piano_transposed = {}
            onsets_piano_transposed = {}
            duration_piano_transposed = {}
            pianoroll_orchestra_transposed = {}
            onsets_orchestra_transposed = {}
            for transposition_semi_tone in range(-self.max_transposition, self.max_transposition + 1):
                # Transpose score
                arr_pair_transposed = {k: v.transpose(transposition_semi_tone) for k, v in arr_pair.items()}

                this_pianoroll_piano, this_onsets_piano, this_durations_piano, _ = score_to_pianoroll(
                    arr_pair_transposed['Piano'],
                    self.subdivision,
                    self.simplify_instrumentation,
                    self.transpose_to_sounding_pitch)
                assert (this_pianoroll_piano.keys() != 'Piano'), 'More than one instrument in the piano score'
                # Quantize piano
                this_quantized_piano_pianoroll = quantize_velocity_pianoroll_frame(this_pianoroll_piano["Piano"],
                                                                                   self.velocity_quantization)

                this_pianoroll_orchestra, this_onsets_orchestra, _, _ = score_to_pianoroll(
                    arr_pair_transposed['Orchestra'],
                    self.subdivision,
                    self.simplify_instrumentation,
                    self.transpose_to_sounding_pitch)

                pianoroll_piano_transposed[transposition_semi_tone] = this_quantized_piano_pianoroll
                onsets_piano_transposed[transposition_semi_tone] = this_onsets_piano
                duration_piano_transposed[transposition_semi_tone] = this_durations_piano
                pianoroll_orchestra_transposed[transposition_semi_tone] = this_pianoroll_orchestra
                onsets_orchestra_transposed[transposition_semi_tone] = this_onsets_orchestra
            ############################################################

            ############################################################
            #  Align (we can use non transposed scores, changes nothing to the alignement
            corresponding_frames, this_scores = self.align_score(arr_pair['Piano'], arr_pair['Orchestra'])
            if self.compute_statistics_flag:
                scores.extend(this_scores)
            # Get corresponding pitch_classes (for statistics)
            pc_piano_list = [e[0][1] for e in corresponding_frames]
            pc_orchestra_list = [e[1][1] for e in corresponding_frames]
            ############################################################

            # Prepare chunks of indices
            chunks_piano_indices, chunks_orchestra_indices = self.prepare_chunk_from_corresponding_frames(
                corresponding_frames)

            for (this_chunk_piano_indices, this_chunk_orchestra_indices) \
                    in zip(chunks_piano_indices, chunks_orchestra_indices):

                total_chunk_counter += 1

                ############################################################
                #  Determine allowed transpositions for this chunk
                # Observe tessitura for each instrument for this chunk. Use non transposed pr of course
                this_min_transposition, this_max_transposition = \
                    self.get_allowed_transpositions_from_pr(pianoroll_piano_transposed[0],
                                                            this_chunk_piano_indices,
                                                            "Piano")
                min_transposition = max(this_min_transposition, -self.max_transposition)
                max_transposition = min(this_max_transposition, self.max_transposition)

                # Use reference tessitura or compute tessitura directly on the files ?
                for instrument_name, pr in pianoroll_orchestra_transposed[0].items():

                    this_min_transposition, this_max_transposition = \
                        self.get_allowed_transpositions_from_pr(pr,
                                                                this_chunk_orchestra_indices,
                                                                instrument_name)
                    if this_min_transposition is not None:
                        min_transposition = max(this_min_transposition, min_transposition)
                        max_transposition = min(this_max_transposition, max_transposition)
                ############################################################

                for transposition_semi_tone in range(min_transposition, max_transposition + 1):

                    total_chunk_counter_augmented += 1

                    avoid_this_chunk = False
                    local_piano_tensor = []
                    local_orchestra_tensor = []

                    # Avoid first and last indices which is simply here for getting the previous and next frame informations
                    for frame_piano, frame_orchestra in zip(this_chunk_piano_indices, this_chunk_orchestra_indices):

                        # Piano encoded vector
                        if (frame_piano is None) and (frame_orchestra is None):
                            # Padding frames
                            piano_t_encoded = self.piano_padding_vector.clone().detach()
                            orchestra_t_encoded = self.orchestra_padding_vector.clone().detach()
                        else:
                            piano_t_encoded = self.pianoroll_to_piano_tensor(
                                pianoroll_piano_transposed[transposition_semi_tone],
                                onsets_piano_transposed[transposition_semi_tone],
                                durations_piano_transposed[transposition_semi_tone],
                                frame_piano)
                            orchestra_t_encoded = self.pianoroll_to_orchestral_tensor(
                                pianoroll_orchestra_transposed[transposition_semi_tone],
                                onsets_orchestra_transposed[transposition_semi_tone],
                                frame_orchestra)

                        if orchestra_t_encoded is None:
                            avoid_this_chunk = True
                            break

                        local_piano_tensor.append(piano_t_encoded)
                        local_orchestra_tensor.append(orchestra_t_encoded)

                    if avoid_this_chunk:
                        missed_chunk_counter += 1
                        continue

                    assert len(local_piano_tensor) == self.sequence_size
                    assert len(local_orchestra_tensor) == self.sequence_size

                    local_piano_tensor = torch.stack(local_piano_tensor)
                    local_orchestra_tensor = torch.stack(local_orchestra_tensor)

                    # append and add batch dimension
                    # cast to int
                    piano_tensor_dataset.append(
                        local_piano_tensor[None, :, :].int())
                    orchestra_tensor_dataset.append(
                        local_orchestra_tensor[None, :, :].int())

            if self.compute_statistics_flag:
                for pc_piano, pc_orchestra in zip(pc_piano_list, pc_orchestra_list):
                    total_frames_counter += 1
                    # Statistics: compare pitch class in orchestra and in piano
                    if pc_piano != pc_orchestra:
                        num_frames_with_different_pitch_class += 1
                        with open(f"{self.compute_statistics_flag}/different_set_pc.txt", "a") as ff:
                            for this_pc in pc_piano:
                                ff.write(f"{this_pc} ")
                            ff.write("// ")
                            for this_pc in pc_orchestra:
                                ff.write(f"{this_pc} ")
                            ff.write("\n")

        piano_tensor_dataset = torch.cat(piano_tensor_dataset, 0)
        orchestra_tensor_dataset = torch.cat(orchestra_tensor_dataset, 0)

        #######################
        if self.compute_statistics_flag:
            # NW statistics
            mean_score = np.mean(scores)
            variance_score = np.var(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            nw_statistics_folder = f"{self.compute_statistics_flag}/nw"
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
            pitch_class_statistics_folder = f"{self.compute_statistics_flag}/pitch_class"
            if os.path.isdir(pitch_class_statistics_folder):
                shutil.rmtree(pitch_class_statistics_folder)
            os.makedirs(pitch_class_statistics_folder)
            # Move different set pc
            shutil.move(f"{self.compute_statistics_flag}/different_set_pc.txt",
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
                                orchestra_tensor_dataset)
        #######################

        print(
            f'Sizes: \n Piano: {piano_tensor_dataset.size()}\n Orchestra: {orchestra_tensor_dataset.size()}\n')
        print(f'Chunks: {total_chunk_counter} with augmentations: {total_chunk_counter_augmented}')
        print(f'Missed chunks: {missed_chunk_counter}  Ratio: {missed_chunk_counter / total_chunk_counter}')
        return dataset

    def get_score_tensor(self, scores, offsets):
        return

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return

    def get_metadata_tensor(self, score):
        return None

    def score_to_list_pc(self, score):
        # Need only the flatten orchestra for aligning
        if score.atSoundingPitch != 'unknown':
            sounding_pitch_score = score.toSoundingPitch()
        score_flat = sounding_pitch_score.flat
        notes_and_chords = score_flat.notes

        list_pc = []
        current_frame_index = 0
        current_set_pc = set()

        for elem in notes_and_chords:
            # Don't consider elements which are not on a subdivision of the beat
            note_start, note_end = quantize_and_filter_music21_element(elem, self.subdivision)
            if note_start is None:
                continue
            assert (note_start >= current_frame_index), "Elements are not sorted by increasing time ?"
            if note_start > current_frame_index:
                # Write in list_pc and move to next
                if len(current_set_pc) > 0:  # Check on length is only for the first iteration
                    list_pc.append((current_frame_index, current_set_pc))
                current_set_pc = set()
                current_frame_index = note_start

            if elem.isNote:
                current_set_pc.add(elem.pitch.pitchClass)
            else:
                current_set_pc = current_set_pc.union(set(elem.pitchClasses))
        # Don't forget last event !
        list_pc.append((current_frame_index, current_set_pc))
        return list_pc

    def align_score(self, piano_score, orchestra_score):
        list_pc_piano = self.score_to_list_pc(piano_score)
        list_pc_orchestra = self.score_to_list_pc(orchestra_score)

        only_pc_piano = [e[1] for e in list_pc_piano]
        only_pc_orchestra = [e[1] for e in list_pc_orchestra]

        corresponding_indices, score_matrix = nw_align.nwalign(only_pc_piano, only_pc_orchestra, gapOpen=-3,
                                                               gapExtend=-1)

        corresponding_frames = [(list_pc_piano[ind_piano], list_pc_orchestra[ind_orchestra])
                                for ind_piano, ind_orchestra in corresponding_indices]

        return corresponding_frames, score_matrix

    ###################################
    # Small helpers for quickly determining which score is orchestra and which one is piano
    def sort_arrangement_pairs(self, arrangement_pair):
        # Find which score is piano and which is orchestral
        if len(self.list_instru_score(arrangement_pair[0])) > len(self.list_instru_score(arrangement_pair[1])):
            return {'Orchestra': arrangement_pair[0], 'Piano': arrangement_pair[1]}
        elif len(self.list_instru_score(arrangement_pair[0])) < len(self.list_instru_score(arrangement_pair[1])):
            return {'Piano': arrangement_pair[0], 'Orchestra': arrangement_pair[1]}
        else:
            raise Exception('The two scores have the same number of instruments')

    def list_instru_score(self, score):
        list_instru = []
        for part in score.parts:
            list_instru.append(part.partName)
        return list_instru

    ###################################

    def get_allowed_transpositions_from_pr(self, pr, frames, instrument_name):
        #  Get min and max pitches
        pr_frames = np.asarray([pr[frame] for frame in frames if frame is not None])
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
        for index_frame in range(1, number_corresponding_frames + 1):
            start_index = index_frame - self.sequence_size
            start_index_truncated = max(0, start_index)
            #  Always add at least one None frame at the beginning (instead of a the real previous frame)
            #  Hence, we avoid the model observe slurs from unseen previous frame
            padding_beginning = start_index_truncated - start_index
            end_index = index_frame
            padding_end = end_index - end_index

            #  Always include a None frame as first frame instead of the real previous frame.
            # This is because we don't want to have slurs from frames that the model cannot observe
            this_piano_chunk = [e[0][0] for e in corresponding_frames[start_index_truncated:end_index]]
            this_orchestra_chunk = [None] + [e[1][0] for e in
                                             corresponding_frames[start_index_truncated:end_index]]

            # Padding
            this_piano_chunk = [None] * padding_beginning + this_piano_chunk + [None] * padding_end
            this_orchestra_chunk = [None] * (padding_beginning - 1) + this_orchestra_chunk + [None] * padding_end
            chunks_piano_indices.append(this_piano_chunk)
            chunks_orchestra_indices.append(this_orchestra_chunk)
        return chunks_piano_indices, chunks_orchestra_indices

    def pianoroll_to_piano_tensor(self, pr, onsets, durations, frame_index):
        piano_encoded = np.zeros((self.number_ticks_piano))
        #  Write one-hot
        for midi_pitch, index_velocity in self.midi_pitch_velocity2index_piano.items():
            this_velocity = pr[frame_index, midi_pitch]
            this_duration = durations[frame_index, midi_pitch]
            index_duration = self.midi_pitch_duration2index_piano[midi_pitch]
            if (this_velocity != 0) and (onsets[frame_index, midi_pitch] == 0):
                piano_encoded[index_velocity] = self.value2oneHot_perPianoToken[index_velocity][SLUR_SYMBOL]
            else:
                piano_encoded[index_velocity] = self.value2oneHot_perPianoToken[index_velocity][this_velocity]
            # Even for SLUR, write duration
            piano_encoded[index_duration] = self.value2oneHot_perPianoToken[index_duration][this_duration]

        piano_tensor = torch.from_numpy(piano_encoded).long()
        return piano_tensor

    def pianoroll_to_orchestral_tensor(self, pianoroll, onsets, frame_index):
        orchestra_encoded = np.zeros((self.number_instruments))
        for instrument_name, indices_instruments in self.instrument2index.items():
            number_of_parts = len(indices_instruments)
            if instrument_name not in pianoroll:
                notes_played = []
            else:
                notes_played = list(np.where(pianoroll[instrument_name][frame_index])[0])
            if len(notes_played) > number_of_parts:
                return None
            # Pad with silences
            notes_played.extend([REST_SYMBOL] * (number_of_parts - len(notes_played)))
            for this_note, this_instrument_index in zip(notes_played, indices_instruments):
                slur_bool = onsets[instrument_name][frame_index, this_note] == 0
                if (this_note != REST_SYMBOL) and slur_bool:
                    orchestra_encoded[this_instrument_index] = self.midi_pitch2index[this_instrument_index][SLUR_SYMBOL]
                else:
                    orchestra_encoded[this_instrument_index] = self.midi_pitch2index[this_instrument_index][this_note]

        orchestra_tensor = torch.from_numpy(orchestra_encoded).long()
        return orchestra_tensor

    def extract_score_tensor_with_padding(self, tensor_score):
        return None

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        return None

    def empty_score_tensor(self, score_length):

        return None

    def random_score_tensor(self, score_length):
        return None

    # def get_offset_duration_from_frames_onsets(self, rhythm):
    #     rhythm = np.asarray(rhythm)
    #     next_rhythm = list(rhythm[1:])
    #     next_rhythm.append(rhythm[-1] + 1)
    #     durations = next_rhythm - rhythm
    #     offsets = [float(e / self.subdivision) for e in rhythm]
    #     durations = [float(e / self.subdivision) for e in durations]
    #     return offsets, durations

    def piano_tensor_to_score(self, tensor_score):

        piano_matrix = tensor_score.numpy()
        length = len(piano_matrix)

        this_part = music21.stream.Part(id='Piano')
        music21_instrument = music21.instrument.fromString('Piano')
        this_part.insert(music21_instrument)

        # Browse pitch dimension first, to deal with sustained notes
        duration_ind = self.midi_pitch2index_piano[DURATION_SYMBOL]
        duration_mapping = self.oneHot2value_perPianoToken[duration_ind]
        for piano_index, pitch in self.index2midi_pitch_piano.items():
            if pitch == DURATION_SYMBOL:
                continue
            current_offset = 0
            offset = 0
            duration = 0
            velocity = 0
            # f = None
            for frame_index in range(length):
                current_velocity = self.oneHot2value_perPianoToken[piano_index][piano_matrix[frame_index, piano_index]]
                current_duration = duration_mapping[piano_matrix[frame_index, duration_ind]]

                #  Need to decide what to do with padding frames here...
                # We decide to write them as a quarter note length silence
                if current_duration == PAD_SYMBOL:
                    current_duration = self.subdivision

                # Write note if current note is not slured
                if current_velocity != SLUR_SYMBOL:
                    #  Write previous frame if it was not a silence
                    if velocity != 0:
                        f = music21.note.Note(pitch)
                        f.volume.velocity = unquantize_velocity(velocity, self.velocity_quantization)
                        f.quarterLength = duration / self.subdivision
                        this_part.insert((offset / self.subdivision), f)
                        # print(f"{pitch} - {offset/self.subdivision} - {duration/self.subdivision} - {unquantize_velocity(velocity, self.velocity_quantization)}")
                    # Reinitialise (note that we don't need to write silences, they are handled by the offset)
                    duration = current_duration
                    offset = current_offset
                    velocity = current_velocity
                elif current_velocity == SLUR_SYMBOL:
                    duration += current_duration

                current_offset += current_duration

            # Don't forget the last note
            if velocity != 0:
                f = music21.note.Note(pitch)
                f.volume.velocity = unquantize_velocity(velocity, self.velocity_quantization)
                f.quarterLength = duration / self.subdivision
                this_part.insert((offset / self.subdivision), f)

        # Very important, if not spread the note of the chord
        this_part_chordified = this_part.chordify()

        # Need the sequence of duration to reconstruct the corresponding orchestration
        list_durations_oneHot = piano_matrix[:, duration_ind]
        list_durations = []
        for e in list_durations_oneHot:
            e_symbol = duration_mapping[e]
            if e_symbol == PAD_SYMBOL:
                list_durations.append(self.subdivision)
            else:
                list_durations.append(e_symbol)

        return this_part_chordified, list_durations

    def orchestra_tensor_to_score(self, tensor_score, durations):
        """

        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :param rhythm:
        :return:
        """
        # (batch, num_parts, notes_encoding)
        orchestra_matrix = tensor_score.numpy()

        if durations is None:
            durations = np.arange(len(tensor_score)) * self.subdivision
        else:
            assert len(tensor_score) == len(durations), "Rhythm vector must be the same length as tensor[0]"

        # First store every in a dict {instrus : [time [notes]]}
        score_dict = {}
        for instrument_index in range(self.number_instruments):
            # Get instrument name
            instrument_name = self.index2instrument[instrument_index]
            if instrument_name not in score_dict:
                score_dict[instrument_name] = []

            duration = 0
            offset = 0
            this_offset = 0
            pitch = None
            for frame_index, this_duration in enumerate(durations):
                this_pitch = self.index2midi_pitch[instrument_index][orchestra_matrix[frame_index, instrument_index]]

                if this_pitch == SLUR_SYMBOL:
                    duration += this_duration
                else:
                    #  Write previous pitch
                    if pitch == PAD_SYMBOL:
                        # Special treatment for PADDING frames
                        score_dict[instrument_name].append((0, offset, self.subdivision))
                    elif pitch != REST_SYMBOL:
                        #  Write previous event
                        score_dict[instrument_name].append((pitch, offset, duration))

                    # Reset values
                    duration = this_duration
                    pitch = this_pitch
                    offset = this_offset

                this_offset += this_duration

            # Last note
            if pitch not in [PAD_SYMBOL, REST_SYMBOL]:
                score_dict[instrument_name].append((pitch, offset, duration))

        #  Batch is used as time in the score
        stream = music21.stream.Stream()
        for instrument_name, elems in score_dict.items():
            this_part = music21.stream.Part(id=instrument_name)
            #  re is for removing underscores in instrument names which raise errors in music21
            if instrument_name == "Cymbal":
                music21_instrument = music21.instrument.Cymbals()
            else:
                music21_instrument = music21.instrument.fromString(re.sub('_', ' ', instrument_name))
            this_part.insert(music21_instrument)

            for elem in elems:
                pitch, offset, duration = elem
                f = music21.note.Note(pitch)
                f.volume.velocity = 60.
                f.quarterLength = duration / self.subdivision
                this_part.insert((offset / self.subdivision), f)

            this_part_chordified = this_part.chordify()
            this_part_chordified.atSoundingPitch = self.transpose_to_sounding_pitch
            stream.append(this_part_chordified)

        return stream

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, writing_dir=None, filepath=None):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{os.getcwd()}/dump"

        # Add padding vectors between each example
        batch_size, time_length, num_features = piano_pianoroll.size()
        piano_with_padding_between_batch = torch.zeros(batch_size, time_length + 1, num_features)
        piano_with_padding_between_batch[:, :time_length] = piano_pianoroll
        piano_with_padding_between_batch[:, time_length] = self.piano_padding_vector
        piano_flat = piano_with_padding_between_batch.view(-1, dataset.number_pitch_piano)
        #
        batch_size, time_length, num_features = orchestra_pianoroll.size()
        orchestra_with_padding_between_batch = torch.zeros(batch_size, time_length + 1, num_features)
        orchestra_with_padding_between_batch[:, :time_length] = orchestra_pianoroll
        orchestra_with_padding_between_batch[:, time_length] = self.orchestra_padding_vector
        orchestra_flat = orchestra_with_padding_between_batch.view(-1, dataset.number_instruments)

        piano_part, durations_piano = self.piano_tensor_to_score(piano_flat)
        orchestra_stream = self.orchestra_tensor_to_score(orchestra_flat, durations_piano)

        piano_part.write(fp=f"{writing_dir}/{filepath}_piano.xml", fmt='musicxml')
        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_orchestra.xml", fmt='musicxml')
        # Both in the same score
        orchestra_stream.append(piano_part)
        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_both.xml", fmt='musicxml')


if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path='/home/leo/Recherche/databases/Orchestration/arrangement_mxml',
        subsets=[
            # 'bouliane',
            # 'imslp',
            # 'liszt_classical_archives',
            # 'hand_picked_Spotify',
            'debug'
        ],
        num_elements=None
    )

    kwargs = {}
    kwargs.update(
        {'name': "arrangement_test",
         'corpus_it_gen': corpus_it_gen,
         'cache_dir': '/home/leo/Recherche/DatasetManager/DatasetManager/dataset_cache',
         'subdivision': 2,
         'sequence_size': 3,
         'velocity_quantization': 4,  # Better if it is divided by 128
         'max_transposition': 3,
         'transpose_to_sounding_pitch': True,
         'compute_statistics_flag': "/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/statistics"
         })

    dataset = ArrangementDataset(**kwargs)
    print(f'Creating {dataset.__repr__()}, '
          f'both tensor dataset and parameters')
    if os.path.exists(dataset.tensor_dataset_filepath):
        os.remove(dataset.tensor_dataset_filepath)
    tensor_dataset = dataset.tensor_dataset

    # Data loaders
    (train_dataloader,
     val_dataloader,
     test_dataloader) = dataset.data_loaders(
        batch_size=16,
        split=(0.85, 0.10),
        DEBUG_BOOL_SHUFFLE=False
    )

    # Visualise a few examples
    number_dump = 20
    writing_dir = f"{os.getcwd()}/dump"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, orchestra_batch = sample_batched
        # Flatten matrices
        # piano_flat = piano_batch.view(-1, dataset.number_pitch_piano)
        # piano_flat_t = piano_flat[dataset.sequence_size - 1::dataset.sequence_size]
        # orchestra_flat = orchestra_batch.view(-1, dataset.number_instruments)
        # orchestra_flat_t = orchestra_flat[dataset.sequence_size - 1::dataset.sequence_size]
        # if i_batch > number_dump:
        #     break
        dataset.visualise_batch(piano_batch, orchestra_batch, writing_dir, filepath=f"{i_batch}_seq")
        # dataset.visualise_batch(piano_flat_t, orchestra_flat_t, writing_dir, filepath=f"{i_batch}_t")
