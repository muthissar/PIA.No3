import json
import os
import re
import shutil

import music21
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

import DatasetManager
from DatasetManager.arrangement.arrangement_helper import quantize_velocity_pianoroll_frame, unquantize_velocity, \
    shift_pr_along_pitch_axis, note_to_midiPitch, score_to_pianoroll, new_events
from DatasetManager.arrangement.instrument_grouping import get_instrument_grouping
from DatasetManager.arrangement.instrumentation import get_instrumentation
from DatasetManager.helpers import REST_SYMBOL, SLUR_SYMBOL, END_SYMBOL, START_SYMBOL, \
    YES_SYMBOL, NO_SYMBOL, PAD_SYMBOL
from DatasetManager.music_dataset import MusicDataset


class OrchestrationDataset(MusicDataset):
    """
    Class for purely orchestral scores
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 subdivision,
                 sequence_size,
                 velocity_quantization,
                 max_transposition,
                 integrate_discretization,
                 transpose_to_sounding_pitch,
                 compute_statistics_flag=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super().__init__()
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.subdivision = subdivision  # We use only on beats notes so far
        assert sequence_size % 2 == 1
        self.sequence_size = sequence_size
        self.velocity_quantization = velocity_quantization
        self.max_transposition = max_transposition
        self.transpose_to_sounding_pitch = transpose_to_sounding_pitch
        self.integrate_discretization = integrate_discretization

        #  For consistency, use arrangement mappings
        dataset_manager_path = os.path.dirname(os.path.realpath(DatasetManager.__file__))
        arrangement_path = f'{dataset_manager_path}/arrangement'
        reference_tessitura_path = f'{arrangement_path}/reference_tessitura.json'
        simplify_instrumentation_path = f'{arrangement_path}/simplify_instrumentation.json'

        dump_folder = f'{os.path.expanduser("~")}/Data/dump'
        self.dump_folder = dump_folder
        self.statistic_folder = self.dump_folder + '/orchestration_bert/statistics'
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
        self.instrument_grouping = get_instrument_grouping()
        self.instrumentation = get_instrumentation()

        # mapping orchestra
        self.index2instrument = {}
        self.instrument2index = {}
        self.index2midi_pitch = {}
        self.midi_pitch2index = {}
        # instruments presence
        self.instruments_presence2index = {}
        self.index2instruments_presence = {}
        self.instrument_presence_name2index = {}
        self.instrument_presence_index2name = {}
        # Dimensions
        self.number_instruments = None
        self.instrument_presence_dim = None

        # Often used vectors, computed in compute_index_dicts
        self.precomputed_vectors_orchestra = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            PAD_SYMBOL: None,
            REST_SYMBOL: None,
        }
        self.precomputed_vectors_orchestra_instruments_presence = {
            PAD_SYMBOL: None
        }

        # Compute statistics slows down the construction of the dataset
        self.compute_statistics_flag = compute_statistics_flag
        return

    def __repr__(self):
        return f'OrchestrationDataset-' \
               f'{self.name}-' \
               f'{self.subdivision}-' \
               f'{self.sequence_size}-' \
               f'{self.velocity_quantization}-' \
               f'{self.max_transposition}'

    def iterator_gen(self):
        return (arrangement_pair for arrangement_pair in self.corpus_it_gen())

    def compute_index_dicts_piano(self, set_midiPitch_per_instrument):
        return

    def load_index_dicts(self):
        # pitch ranges
        set_midiPitch_per_instrument = {}
        instrument_name_list = list(self.instrumentation.keys())
        instrument_name_list.append("Piano")
        for instrument_name in instrument_name_list:
            lowest_note, highest_note = self.reference_tessitura[instrument_name]
            lowest_pitch = note_to_midiPitch(lowest_note)
            highest_pitch = note_to_midiPitch(highest_note)
            set_pitches = set(range(lowest_pitch, highest_pitch + 1))
            set_midiPitch_per_instrument[instrument_name] = set_pitches

        if self.compute_statistics_flag:
            with open(f"{self.statistic_folder}/note_frequency_per_instrument", "w") as ff:
                for instrument_name, set_pitch_class in set_midiPitch_per_instrument.items():
                    ff.write(f"# {instrument_name}: \n")
                    for pc in set_pitch_class:
                        ff.write(f"   {pc}\n")

        self.compute_index_dicts_piano(set_midiPitch_per_instrument)
        self.compute_index_dicts_orchestra(set_midiPitch_per_instrument)

        return

    def compute_index_dicts_orchestra(self, set_midiPitch_per_instrument):
        ############################################################
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
            #  Pad
            index += 1
            midi_pitch2index_per_instrument[instrument_name][PAD_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = PAD_SYMBOL
            # Start
            index += 1
            midi_pitch2index_per_instrument[instrument_name][START_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = START_SYMBOL
            # End
            index += 1
            midi_pitch2index_per_instrument[instrument_name][END_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = END_SYMBOL

        # Print instruments avoided
        print("Instruments not used")
        for instrument_name in midi_pitch2index_per_instrument.keys():
            if self.instrumentation[instrument_name] == 0:
                print(f'# {instrument_name}')

        # Mapping instruments <-> indices
        index_counter = 0
        counter_instrument_presence = 0
        for instrument_name, number_instruments in self.instrumentation.items():
            if instrument_name == "Piano":
                continue
            #  Check if instrument appears in the dataset
            if instrument_name not in midi_pitch2index_per_instrument.keys():
                continue
            #  Don't use instruments which are assigned 0 voices
            if number_instruments == 0:
                continue

            self.instrument2index[instrument_name] = list(range(index_counter, index_counter + number_instruments))
            for ind in range(index_counter, index_counter + number_instruments):
                self.index2instrument[ind] = instrument_name
            index_counter += number_instruments

            self.instrument_presence_name2index[instrument_name] = counter_instrument_presence
            self.instrument_presence_index2name[counter_instrument_presence] = instrument_name
            counter_instrument_presence += 1

        # Mapping pitch <-> index per voice (that's the one we'll use, easier to manipulate when training)
        for instrument_name, instrument_indices in self.instrument2index.items():
            for instrument_index in instrument_indices:
                self.midi_pitch2index[instrument_index] = midi_pitch2index_per_instrument[instrument_name]
                self.index2midi_pitch[instrument_index] = index2midi_pitch_per_instrument[instrument_name]
        ############################################################
        ############################################################

        ############################################################
        ############################################################
        # Encoding for orchestra presence
        # Same mapping for all instruments
        #  Unknown symbol is used for dropout during training, and also when generating if you don't want to
        # hard constrain the presence/absence of a note
        self.instruments_presence2index = {
            NO_SYMBOL: 0,
            YES_SYMBOL: 1,
            PAD_SYMBOL: 2
        }
        self.index2instruments_presence = {}
        for k, v in self.instruments_presence2index.items():
            self.index2instruments_presence[v] = k
        ############################################################
        ############################################################

        self.number_instruments = len(self.midi_pitch2index)
        self.instrument_presence_dim = len(self.instrument_presence_index2name)

        orchestra_start_vector = []
        orchestra_end_vector = []
        orchestra_padding_vector = []
        orchestra_rest_vector = []
        for instru_ind, mapping in self.midi_pitch2index.items():
            orchestra_start_vector.append(mapping[START_SYMBOL])
            orchestra_end_vector.append(mapping[END_SYMBOL])
            orchestra_padding_vector.append(mapping[PAD_SYMBOL])
            orchestra_rest_vector.append(mapping[REST_SYMBOL])
        self.precomputed_vectors_orchestra[START_SYMBOL] = torch.from_numpy(np.asarray(orchestra_start_vector)).long()
        self.precomputed_vectors_orchestra[END_SYMBOL] = torch.from_numpy(np.asarray(orchestra_end_vector)).long()
        self.precomputed_vectors_orchestra[PAD_SYMBOL] = torch.from_numpy(np.asarray(orchestra_padding_vector)).long()
        self.precomputed_vectors_orchestra[REST_SYMBOL] = torch.from_numpy(np.asarray(orchestra_rest_vector)).long()
        #
        unknown_vector = np.ones((self.instrument_presence_dim)) * self.instruments_presence2index[PAD_SYMBOL]
        self.precomputed_vectors_orchestra_instruments_presence[PAD_SYMBOL] = torch.from_numpy(unknown_vector).long()
        ############################################################
        ############################################################

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
        orchestra_tensor_dataset = []
        orchestra_instruments_presence_tensor_dataset = []

        # Iterate over files
        for score in tqdm(self.iterator_gen()):

            try:
                pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(
                    score=score['Orchestra'],
                    subdivision=self.subdivision,
                    simplify_instrumentation=self.simplify_instrumentation,
                    instrument_grouping=self.instrument_grouping,
                    transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
                    integrate_discretization=self.integrate_discretization,
                    binarize=False,
                )
            except:
                with open('dump/shit_files.txt', 'a') as ff:
                    ff.write(f'{score["name"]}\n')

            events = new_events(pianoroll_orchestra, onsets_orchestra)

            chunks = self.prepare_chunks(events)

            # First get non transposed score
            transposition_semi_tone = 0
            minimum_transposition_allowed = None
            maximum_transposition_allowed = None
            minimum_transposition_allowed, maximum_transposition_allowed, \
            orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
            total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                self.transpose_loop_iteration(pianoroll_orchestra=pianoroll_orchestra,
                                              onsets_orchestra=onsets_orchestra,
                                              transposition_semi_tone=0,
                                              chunks=chunks,
                                              minimum_transposition_allowed=minimum_transposition_allowed,
                                              maximum_transposition_allowed=maximum_transposition_allowed,
                                              orchestra_tensor_dataset=orchestra_tensor_dataset,
                                              orchestra_instruments_presence_tensor_dataset=orchestra_instruments_presence_tensor_dataset,
                                              total_chunk_counter=total_chunk_counter,
                                              too_many_instruments_frame=too_many_instruments_frame,
                                              impossible_transposition=impossible_transposition)

            for transposition_semi_tone in range(-self.max_transposition, self.max_transposition + 1):
                if transposition_semi_tone == 0:
                    continue
                _, _, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
                total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                    self.transpose_loop_iteration(pianoroll_orchestra=pianoroll_orchestra,
                                                  onsets_orchestra=onsets_orchestra,
                                                  transposition_semi_tone=transposition_semi_tone,
                                                  chunks=chunks,
                                                  minimum_transposition_allowed=minimum_transposition_allowed,
                                                  maximum_transposition_allowed=maximum_transposition_allowed,
                                                  orchestra_tensor_dataset=orchestra_tensor_dataset,
                                                  orchestra_instruments_presence_tensor_dataset=orchestra_instruments_presence_tensor_dataset,
                                                  total_chunk_counter=total_chunk_counter,
                                                  too_many_instruments_frame=too_many_instruments_frame,
                                                  impossible_transposition=impossible_transposition)

        orchestra_tensor_dataset = torch.cat(orchestra_tensor_dataset, 0)
        orchestra_instruments_presence_tensor_dataset = torch.cat(orchestra_instruments_presence_tensor_dataset, 0)

        dataset = TensorDataset(orchestra_tensor_dataset,
                                orchestra_instruments_presence_tensor_dataset)

        print(
            f'### Sizes: \n'
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

    def prepare_chunks(self, events):
        chunks = []
        number_frames = len(events)
        for index_frame in range(0, number_frames):
            # if we consider the time in the middle is the one of interest, we must pad half of seq size at the
            # beginning and half at the end
            start_index = index_frame - (self.sequence_size - 1) // 2
            start_index_truncated = max(0, start_index)
            #  Always add at least one None frame at the beginning (instead of a the real previous frame)
            #  Hence, we avoid the model observe slurs from unseen previous frame
            padding_beginning = start_index_truncated - start_index
            end_index = index_frame + (self.sequence_size - 1) // 2
            end_index_truncated = min(number_frames, end_index)
            padding_end = max(0, end_index - number_frames + 1)

            this_chunk = events[start_index_truncated:end_index_truncated + 1]

            # Padding
            if padding_beginning == 0:
                prepend_vector = []
            else:
                prepend_vector = (padding_beginning - 1) * [PAD_SYMBOL] + [START_SYMBOL]

            if padding_end == 0:
                append_vector = []
            else:
                append_vector = [END_SYMBOL] + (padding_end - 1) * [PAD_SYMBOL]

            this_chunk = prepend_vector + this_chunk + append_vector
            chunks.append(this_chunk)

        return chunks

    def transpose_loop_iteration(self,
                                 pianoroll_orchestra,
                                 onsets_orchestra,
                                 transposition_semi_tone,
                                 chunks,
                                 minimum_transposition_allowed,
                                 maximum_transposition_allowed,
                                 orchestra_tensor_dataset,
                                 orchestra_instruments_presence_tensor_dataset,
                                 total_chunk_counter,
                                 too_many_instruments_frame,
                                 impossible_transposition):

        ############################################################
        # Transpose pianorolls
        this_pr_orchestra = {}
        this_onsets_orchestra = {}
        for instrument_name in pianoroll_orchestra.keys():
            # Pr
            pr = pianoroll_orchestra[instrument_name]
            shifted_pr = shift_pr_along_pitch_axis(pr, transposition_semi_tone)
            this_pr_orchestra[instrument_name] = shifted_pr
            # Onsets
            onsets = onsets_orchestra[instrument_name]
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

        for chunk_index in range(len(chunks)):
            this_chunk_orchestra_indices = chunks[chunk_index]
            avoid_this_chunk = False
            total_chunk_counter += 1

            ############################################################
            if build_allowed_transposition_flag:
                min_transposition = -self.max_transposition
                max_transposition = self.max_transposition

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
            local_orchestra_tensor = []
            local_orchestra_instruments_presence_tensor = []
            previous_notes_orchestra = None
            for frame_orchestra in this_chunk_orchestra_indices:
                # Piano encoded vector
                if frame_orchestra in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                    orchestra_t_encoded = self.precomputed_vectors_orchestra[frame_orchestra].clone().detach()
                    orchestra_instruments_presence_t_encoded = \
                        self.precomputed_vectors_orchestra_instruments_presence[PAD_SYMBOL].clone().detach()
                else:
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

                local_orchestra_tensor.append(orchestra_t_encoded)
                local_orchestra_instruments_presence_tensor.append(orchestra_instruments_presence_t_encoded)
            ############################################################

            if avoid_this_chunk:
                too_many_instruments_frame += 1
                continue

            assert len(local_orchestra_tensor) == self.sequence_size

            local_orchestra_tensor = torch.stack(local_orchestra_tensor)
            local_orchestra_instruments_presence_tensor = torch.stack(local_orchestra_instruments_presence_tensor)

            orchestra_tensor_dataset.append(
                local_orchestra_tensor[None, :, :].int())
            orchestra_instruments_presence_tensor_dataset.append(
                local_orchestra_instruments_presence_tensor[None, :, :].int())

        return minimum_transposition_allowed, maximum_transposition_allowed, \
               orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
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
                    # print(f'OOR: {instrument_name} - {note}')
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

    def piano_tensor_to_score(self, tensor_score, durations=None, writing_tempo='adagio', subdivision=None):

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

        # Tempo
        # t = music21.tempo.MetronomeMark(writing_tempo)
        # this_part.insert(0, t)

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
                        else:
                            f = music21.note.Rest()
                            f.quarterLength = duration / subdivision
                            this_part.insert((offset / subdivision), f)
                    duration = current_duration
                    offset = current_offset
                    velocity = current_velocity
                elif current_velocity == SLUR_SYMBOL:
                    duration += current_duration

                current_offset += current_duration

            # Don't forget the last note
            if velocity not in [REST_SYMBOL]:
                f = music21.note.Note(pitch)
                f.volume.velocity = unquantize_velocity(velocity, self.velocity_quantization)
                f.quarterLength = duration / subdivision
                this_part.insert((offset / subdivision), f)

        return this_part

    def orchestra_tensor_to_score(self, tensor_score, durations=None, writing_tempo="adagio", subdivision=None):
        """

        :param durations:
        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :return:
        """
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

        #  Batch is used as time in the score
        stream = music21.stream.Stream()

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

            # Tempo
            # t = music21.tempo.MetronomeMark(writing_tempo)
            # this_part.insert(0, t)

            if elems == []:
                f = music21.note.Rest()
                f.quarterLength = total_duration_ql
                this_part.insert(0, f)
            else:
                #  Sort by offset time (not sure it's very useful, more for debugging purposes)
                elems = sorted(elems, key=lambda e: e[1])
                for elem in elems:
                    pitch, offset, duration = elem
                    f = music21.note.Note(pitch)
                    f.volume.velocity = 60.
                    f.quarterLength = duration / subdivision
                    this_part.insert((offset / subdivision), f)

            this_part.atSoundingPitch = self.transpose_to_sounding_pitch
            stream.append(this_part)

        return stream

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano=None, writing_dir=None,
                        filepath=None, writing_tempo='adagio', subdivision=None):
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

        piano_part = self.piano_tensor_to_score(piano_flat, durations_piano, writing_tempo=writing_tempo,
                                                subdivision=subdivision)
        orchestra_stream = self.orchestra_tensor_to_score(orchestra_flat, durations_piano, writing_tempo=writing_tempo,
                                                          subdivision=subdivision)

        piano_part.write(fp=f"{writing_dir}/{filepath}_piano.mid", fmt='midi')
        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_orchestra.mid", fmt='midi')
        # Both in the same score
        orchestra_stream.append(piano_part)
        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_both.mid", fmt='midi')

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[],
                                 subdivision=None):
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
                    orchestra_init[:, instrument_index] = self.midi_pitch2index[instrument_index][MASK_SYMBOL]

        # Start and end symbol at the beginning and end
        orchestra_init[:context_length - 1] = self.precomputed_vectors_orchestra[PAD_SYMBOL]
        orchestra_init[context_length - 1] = self.precomputed_vectors_orchestra[START_SYMBOL]
        orchestra_init[-context_length] = self.precomputed_vectors_orchestra[END_SYMBOL]
        orchestra_init[-context_length:] = self.precomputed_vectors_orchestra[PAD_SYMBOL]
        return orchestra_silences, orchestra_unknown, instruments_presence, orchestra_init
