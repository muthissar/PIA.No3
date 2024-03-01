import json
import os
import re
import shutil

import torch
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import music21
import numpy as np

from DatasetManager.arrangement.instrumentation import get_instrumentation
from DatasetManager.helpers import PAD_SYMBOL, REST_SYMBOL, SLUR_SYMBOL, END_SYMBOL, START_SYMBOL
from DatasetManager.music_dataset import MusicDataset
import DatasetManager.arrangement.nw_align as nw_align

from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll, quantize_and_filter_music21_element, \
    quantize_velocity_pianoroll_frame, unquantize_velocity, shift_pr_along_pitch_axis


class ArrangementPianorollDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 subdivision=2,
                 sequence_size=3,
                 velocity_quantization=8,
                 max_transposition=3,
                 transpose_to_sounding_pitch=True,
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
        super(ArrangementPianorollDataset, self).__init__()
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.subdivision = subdivision  # We use only on beats notes so far
        assert sequence_size % 2 == 1
        self.sequence_size = sequence_size
        self.velocity_quantization = velocity_quantization
        self.max_transposition = max_transposition
        self.transpose_to_sounding_pitch = transpose_to_sounding_pitch

        dump_folder = f'{os.path.expanduser("~")}/Data/dump'
        reference_tessitura_path = 'reference_tessitura.json'
        simplify_instrumentation_path = 'simplify_instrumentation.json'
        self.dump_folder = dump_folder
        self.statistic_folder = self.dump_folder + '/arrangement_pianoroll/statistics'
        if os.path.isdir(self.statistic_folder):
            shutil.rmtree(self.statistic_folder)
        os.makedirs(self.statistic_folder)

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

        # Compute statistics slows down the construction of the dataset
        self.compute_statistics_flag = compute_statistics_flag
        return

    def __repr__(self):
        return f'ArrangementPR-' \
               f'{self.name}-' \
               f'{self.subdivision}-' \
               f'{self.sequence_size}-' \
               f'{self.velocity_quantization}-' \
               f'{self.max_transposition}'

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
                                                                  None,
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
            with open(f"{self.statistic_folder}/pc_frequency_per_instrument", "w") as ff:
                for instrument_name, set_pitch_class in set_midiPitch_per_instrument.items():
                    ff.write(f"# {instrument_name}: \n")
                    for pc in set_pitch_class:
                        ff.write(f"   {pc}\n")

        index_counter = 0
        for instrument_name, set_midiPitch in set_midiPitch_per_instrument.items():
            if instrument_name == 'Piano':
                min_pitch = min(set_midiPitch)
                max_pitch = max(set_midiPitch) + 1  #  +1 is for creating intervals
                length = max_pitch - min_pitch
                self.observed_tessitura["Piano"] = {
                    "pitch_min": min_pitch,
                    "pitch_max": max_pitch,
                    "index_min": 0,
                    "index_max": length
                }
            else:
                min_pitch = min(set_midiPitch)
                max_pitch = max(set_midiPitch) + 1  #  +1 is for creating intervals
                length = max_pitch - min_pitch
                min_index = index_counter
                max_index = index_counter + length
                index_counter += length
                self.observed_tessitura[instrument_name] = {
                    "pitch_min": min_pitch,
                    "pitch_max": max_pitch,
                    "index_min": min_index,
                    "index_max": max_index
                }
        return

    def make_tensor_dataset(self, frame_orchestra=None):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        self.compute_index_dicts()

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

        # Iterate over files
        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

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

            # Compute original pianorolls
            pianoroll_piano, onsets_piano, _ = score_to_pianoroll(
                arr_pair['Piano'],
                self.subdivision,
                None,
                self.transpose_to_sounding_pitch)

            # Quantize piano
            pr_piano = quantize_velocity_pianoroll_frame(pianoroll_piano["Piano"],
                                                         self.velocity_quantization)
            onsets_piano = onsets_piano["Piano"]

            pr_orchestra, onsets_orchestra, _ = score_to_pianoroll(
                arr_pair['Orchestra'],
                self.subdivision,
                self.simplify_instrumentation,
                self.transpose_to_sounding_pitch)

            pr_pair = {"Piano": pr_piano, "Orchestra": pr_orchestra}
            onsets_pair = {"Piano": onsets_piano, "Orchestra": onsets_orchestra}

            # First get non transposed score
            transposition_semi_tone = 0
            minimum_transpositions_allowed = None
            maximum_transpositions_allowed = None
            minimum_transpositions_allowed, maximum_transpositions_allowed, \
            piano_tensor_dataset, orchestra_tensor_dataset, \
            total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                self.transpose_loop_iteration(pr_pair, onsets_pair, transposition_semi_tone,
                                              chunks_piano_indices, chunks_orchestra_indices,
                                              minimum_transpositions_allowed, maximum_transpositions_allowed,
                                              piano_tensor_dataset, orchestra_tensor_dataset,
                                              total_chunk_counter, too_many_instruments_frame, impossible_transposition)

            for transposition_semi_tone in range(-self.max_transposition, self.max_transposition + 1):
                if transposition_semi_tone == 0:
                    continue
                _, _, piano_tensor_dataset, orchestra_tensor_dataset, total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                    self.transpose_loop_iteration(pr_pair, onsets_pair, transposition_semi_tone,
                                                  chunks_piano_indices, chunks_orchestra_indices,
                                                  minimum_transpositions_allowed, maximum_transpositions_allowed,
                                                  piano_tensor_dataset, orchestra_tensor_dataset,
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
                                orchestra_tensor_dataset)
        #######################

        print(
            f'Sizes: \n Piano: {piano_tensor_dataset.size()}\n Orchestra: {orchestra_tensor_dataset.size()}\n')
        print(
            f'Chunks: {total_chunk_counter}\nToo many instru chunks: {too_many_instruments_frame}\nImpossible transpo: {impossible_transposition}')
        return dataset

    def get_score_tensor(self, scores, offsets):
        return

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return

    def get_metadata_tensor(self, score):
        return None

    def score_to_list_pc(self, score):

        list_pc = []
        current_frame_index = 0
        current_set_pc = set()

        if self.transpose_to_sounding_pitch:
            if score.atSoundingPitch != 'unknown':
                score_soundingPitch = score.toSoundingPitch()
        else:
            score_soundingPitch = score

        # TODO Filter out the parts associated to remove ?? Did not raise issue yet, but could be some day
        elements_iterator = score_soundingPitch.flat.notes

        for elem in elements_iterator:
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
        pr_frames = np.asarray(
            [pr[frame] for frame in frames if frame not in [REST_SYMBOL, PAD_SYMBOL, START_SYMBOL, END_SYMBOL]])
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
            # if we consider the time in the middle is the one of interest, we must pad half of seq size at the beginning and half at the end
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
            this_piano_chunk = [REST_SYMBOL] * padding_beginning + this_piano_chunk + [REST_SYMBOL] * padding_end
            this_orchestra_chunk = [START_SYMBOL] * padding_beginning + this_orchestra_chunk + [
                END_SYMBOL] * padding_end
            chunks_piano_indices.append(this_piano_chunk)
            chunks_orchestra_indices.append(this_orchestra_chunk)
        return chunks_piano_indices, chunks_orchestra_indices

    def transpose_loop_iteration(self, pianorolls_pair, onsets_pair, transposition_semi_tone,
                                 chunks_piano_indices, chunks_orchestra_indices,
                                 minimum_transposition_allowed, maximum_transposition_allowed,
                                 piano_tensor_dataset, orchestra_tensor_dataset,
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
            for frame_piano, frame_orchestra in zip(this_chunk_piano_indices, this_chunk_orchestra_indices):
                # Piano encoded vector
                if (frame_piano == REST_SYMBOL) and (frame_orchestra in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]):
                    piano_t_encoded = self.precomputed_vectors_piano[frame_piano].clone().detach()
                    orchestra_t_encoded = self.precomputed_vectors_orchestra[frame_orchestra].clone().detach()
                else:
                    piano_t_encoded = self.pianoroll_to_piano_tensor(
                        this_pr_piano,
                        this_onsets_piano,
                        frame_piano)
                    orchestra_t_encoded = self.pianoroll_to_orchestral_tensor(
                        this_pr_orchestra,
                        this_onsets_orchestra,
                        frame_orchestra)

                if orchestra_t_encoded is None:
                    avoid_this_chunk = True
                    break

                local_piano_tensor.append(piano_t_encoded)
                local_orchestra_tensor.append(orchestra_t_encoded)
            ############################################################

            if avoid_this_chunk:
                too_many_instruments_frame += 1
                continue

            assert len(local_piano_tensor) == self.sequence_size
            assert len(local_orchestra_tensor) == self.sequence_size

            local_piano_tensor = torch.stack(local_piano_tensor)
            local_orchestra_tensor = torch.stack(local_orchestra_tensor)

            piano_tensor_dataset.append(
                local_piano_tensor[None, :, :].int())
            orchestra_tensor_dataset.append(
                local_orchestra_tensor[None, :, :].int())

        return minimum_transposition_allowed, maximum_transposition_allowed, \
               piano_tensor_dataset, orchestra_tensor_dataset, \
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
                slur_bool = False
                if this_note != REST_SYMBOL:
                    slur_bool = (onsets[instrument_name][frame_index, this_note] == 0)

                if slur_bool:
                    orchestra_encoded[this_instrument_index] = self.midi_pitch2index[this_instrument_index][SLUR_SYMBOL]
                else:
                    orchestra_encoded[this_instrument_index] = self.midi_pitch2index[this_instrument_index][
                        this_note]
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

    def piano_tensor_to_score(self, tensor_score, durations=None):

        piano_matrix = tensor_score.numpy()
        length = len(piano_matrix)

        if durations is None:
            durations = np.ones((length)) * self.subdivision
        assert length == len(durations)

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
                        if velocity != REST_SYMBOL:
                            f = music21.note.Note(pitch)
                            f.volume.velocity = unquantize_velocity(velocity, self.velocity_quantization)
                            f.quarterLength = duration / self.subdivision
                            this_part.insert((offset / self.subdivision), f)
                            # print(f"{pitch} - {offset/self.subdivision} - {duration/self.subdivision} - {unquantize_velocity(velocity, self.velocity_quantization)}")
                        # Reinitialise (note that we don't need to write silences, they are handled by the offset)
                        else:
                            f = music21.note.Rest()
                            f.quarterLength = duration / self.subdivision
                            this_part.insert((offset / self.subdivision), f)
                    duration = current_duration
                    offset = current_offset
                    velocity = current_velocity
                elif current_velocity == SLUR_SYMBOL:
                    duration += current_duration

                current_offset += current_duration

            # Don't forget the last note
            if velocity != REST_SYMBOL:
                f = music21.note.Note(pitch)
                f.volume.velocity = unquantize_velocity(velocity, self.velocity_quantization)
                f.quarterLength = duration / self.subdivision
                this_part.insert((offset / self.subdivision), f)

        # Very important, if not spread the note of the chord
        this_part_chordified = this_part.chordify()

        return this_part_chordified

    def orchestra_tensor_to_score(self, tensor_score, durations=None):
        """

        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :param rhythm:
        :return:
        """
        # (batch, num_parts, notes_encoding)
        orchestra_matrix = tensor_score.numpy()
        length = len(orchestra_matrix)

        if durations is None:
            durations = np.ones((length)) * self.subdivision
        else:
            assert length == len(durations), "Rhythm vector must be the same length as tensor[0]"

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
                    if pitch is not None:
                        #  Write previous pitch
                        if pitch == PAD_SYMBOL:
                            # Special treatment for PADDING frames
                            score_dict[instrument_name].append((0, offset, self.subdivision))  # In fact, pitch 0 = C4
                        elif pitch == START_SYMBOL:
                            # Special treatment for PADDING frames
                            score_dict[instrument_name].append((0, offset, self.subdivision))
                            score_dict[instrument_name].append((6, offset, self.subdivision))
                        elif pitch == END_SYMBOL:
                            # Special treatment for PADDING frames
                            score_dict[instrument_name].append((6, offset, self.subdivision))
                            score_dict[instrument_name].append((7, offset, self.subdivision))
                        elif pitch != REST_SYMBOL:
                            #  Write previous event
                            score_dict[instrument_name].append((pitch, offset, duration))

                    # Reset values
                    duration = this_duration
                    pitch = this_pitch
                    offset = this_offset

                this_offset += this_duration

            # Last note
            if pitch != REST_SYMBOL:
                if pitch == PAD_SYMBOL:
                    score_dict[instrument_name].append((0, offset, duration))
                elif pitch == START_SYMBOL:
                    # Special treatment for PADDING frames
                    score_dict[instrument_name].append((0, offset, self.subdivision))
                    score_dict[instrument_name].append((6, offset, self.subdivision))
                elif pitch == END_SYMBOL:
                    # Special treatment for PADDING frames
                    score_dict[instrument_name].append((6, offset, self.subdivision))
                    score_dict[instrument_name].append((7, offset, self.subdivision))
                elif pitch != REST_SYMBOL:
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

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano=None, writing_dir=None,
                        filepath=None):
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
            orchestra_with_padding_between_batch[:, time_length] = self.precomputed_vectors_orchestra[PAD_SYMBOL]
            orchestra_flat = orchestra_with_padding_between_batch.view(-1, self.number_instruments)

        piano_part = self.piano_tensor_to_score(piano_flat, durations_piano)
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
        {'name': "arrangement_PR_SHIT",
         'corpus_it_gen': corpus_it_gen,
         'cache_dir': '/home/leo/Recherche/DatasetManager/DatasetManager/dataset_cache',
         'subdivision': 2,
         'sequence_size': 5,
         'max_transposition': 6,
         'transpose_to_sounding_pitch': True,
         'compute_statistics_flag': True
         })

    dataset = ArrangementPianorollDataset(**kwargs)
    print(f'Creating {dataset.__repr__()}, '
          f'both tensor dataset and parameters')
    # if os.path.exists(dataset.tensor_dataset_filepath):
    #     os.remove(dataset.tensor_dataset_filepath)
    tensor_dataset = dataset.tensor_dataset

    # Data loaders
    (train_dataloader,
     val_dataloader,
     test_dataloader) = dataset.data_loaders(
        batch_size=8,
        split=(0.85, 0.10),
        DEBUG_BOOL_SHUFFLE=False
    )

    # Visualise a few examples
    number_dump = 20
    writing_dir = f"{dataset.dump_folder}/arrangement/writing"
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
        dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir, filepath=f"{i_batch}_seq")
        # dataset.visualise_batch(piano_flat_t, orchestra_flat_t, writing_dir, filepath=f"{i_batch}_t")
