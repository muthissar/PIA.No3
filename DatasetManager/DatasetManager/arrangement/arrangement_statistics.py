import copy
import csv
import os
import shutil

import DatasetManager
from DatasetManager.arrangement.instrument_grouping import get_instrument_grouping
from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator, note_to_midiPitch, \
    separate_instruments_names, OrchestraIteratorGenerator, score_to_pianoroll
import music21
import numpy as np
import json
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


class ComputeStatistics:
    def __init__(self, score_iterator, subdivision, savefolder_name, sounding_pitch_boolean=False):

        dataset_manager_path = os.path.dirname(os.path.realpath(DatasetManager.__file__))
        database_path = f'{dataset_manager_path}/databases'
        dump_folder = f'{dataset_manager_path}/dump'
        simplify_instrumentation_path = f'{dataset_manager_path}/arrangement/simplify_instrumentation.json'

        #  Dump folder
        self.dump_folder = dump_folder
        self.savefolder_name = f'{self.dump_folder}/{savefolder_name}/statistics'
        if os.path.isdir(self.savefolder_name):
            shutil.rmtree(self.savefolder_name)
        os.makedirs(self.savefolder_name)

        self.num_bins = 30

        #  Simplify instrumentation
        with open(simplify_instrumentation_path, 'r') as ff:
            self.simplify_instrumentation = json.load(ff)
        self.instrument_grouping = get_instrument_grouping()

        #  Histogram with range of notes per instrument
        self.tessitura = dict()

        # Histogram with number of simultaneous notes
        self.simultaneous_notes = dict()

        # Histogram with the number of simultaneous notes per instrument
        self.stat_dict = dict()

        #  Other stuffs used for parsing
        self.score_iterator = score_iterator
        self.subdivision = subdivision
        self.sounding_pitch_boolean = sounding_pitch_boolean

        #  Write out paths here
        self.simultaneous_notes_details_path = f'{self.savefolder_name}/simultaneous_notes_details.txt'
        open(self.simultaneous_notes_details_path, 'w').close()
        if self.sounding_pitch_boolean:
            self.tessitura_path = f'{self.savefolder_name}/tessitura_sounding'
        else:
            self.tessitura_path = f'{self.savefolder_name}/tessitura'
        if os.path.isdir(self.tessitura_path):
            shutil.rmtree(self.tessitura_path)
        os.makedirs(self.tessitura_path)

        return

    def get_statistics(self):
        for arrangement_pair in self.score_iterator():
            if arrangement_pair is None:
                continue
            # Orchestra scores
            self.histogram_tessitura(arrangement_pair['Orchestra'])
            self.counting_simultaneous_notes(arrangement_pair['Orchestra'], True)
            self.counting_simultaneous_notes(arrangement_pair['Piano'], False)

        # Get reference tessitura for comparing when plotting
        with open('reference_tessitura.json') as ff:
            reference_tessitura = json.load(ff)
        reference_tessitura = {
        k: (note_to_midiPitch(music21.note.Note(v[0])), note_to_midiPitch(music21.note.Note(v[1]))) for
        k, v in reference_tessitura.items()}

        stats = []
        this_stats = {}
        for instrument_name, histogram in self.tessitura.items():
            # Plot histogram for each instrument
            x = range(128)
            y = list(histogram.astype(int))
            plt.clf()
            plt.bar(x, y)
            plt.xlabel('Notes', fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
            plt.title(instrument_name, fontsize=10)
            plt.xticks(np.arange(0, 128, 1))
            # Add reference tessitura
            if instrument_name != "Remove":
                min_ref, max_ref = reference_tessitura[instrument_name]
            else:
                min_ref = 0
                max_ref = 128
            x = range(min_ref, max_ref)
            y = [0 for _ in range(len(x))]
            plt.plot(x, y, 'ro')
            plt.savefig(f'{self.tessitura_path}/{instrument_name}_tessitura.pdf')

            # Write stats in txt file
            total_num_notes = histogram.sum()
            non_zero_indices = np.nonzero(histogram)[0]
            lowest_pitch = non_zero_indices.min()
            highest_pitch = non_zero_indices.max()
            this_stats = {
                'instrument_name': instrument_name,
                'total_num_notes': total_num_notes,
                'lowest_pitch': lowest_pitch,
                'highest_pitch': highest_pitch
            }
            stats.append(this_stats)

        # Write number of co-occuring notes
        with open(f'{self.savefolder_name}/simultaneous_notes.txt', 'w') as ff:
            for instrument_name, simultaneous_counter in self.simultaneous_notes.items():
                ff.write(f"## {instrument_name}\n")
                for ind, simultaneous_occurences in enumerate(list(simultaneous_counter)):
                    ff.write('  {:d} : {:d}\n'.format(ind, int(simultaneous_occurences)))

        with open(f'{self.savefolder_name}/statistics.csv', 'w') as ff:
            fieldnames = this_stats.keys()
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for this_stats in stats:
                writer.writerow(this_stats)

    def histogram_tessitura(self, score):
        # Transpose to sounding pitch ?
        if self.sounding_pitch_boolean:
            if score.atSoundingPitch != 'unknown':
                score_processed = score.toSoundingPitch()
        else:
            score_processed = score

        for part in score_processed.parts:
            instrument_names = [self.instrument_grouping[e] for e in
                                separate_instruments_names(self.simplify_instrumentation[part.partName])]
            for instrument_name in instrument_names:
                part_flat = part.flat
                histogram = music21.graph.plot.HistogramPitchSpace(part_flat)
                histogram.extractData()
                histogram_data = histogram.data
                for (pitch, frequency, _) in histogram_data:
                    if instrument_name not in self.tessitura:
                        self.tessitura[instrument_name] = np.zeros((128,))
                    self.tessitura[instrument_name][int(pitch)] += frequency
        return

    def counting_simultaneous_notes(self, score, orchestra_flag):

        if orchestra_flag:
            pr, onsets, _ = score_to_pianoroll(
                score,
                self.subdivision,
                self.simplify_instrumentation,
                self.instrument_grouping,
                self.sounding_pitch_boolean)
        else:
            pr, onsets, _ = score_to_pianoroll(
                score,
                self.subdivision,
                None,
                self.instrument_grouping,
                self.sounding_pitch_boolean)

        with open(self.simultaneous_notes_details_path, 'a') as ff:
            ff.write(f'##### {score.filePath}\n')

        for instrument_name, this_pr in pr.items():
            # binarize
            this_pr_bin = np.where(this_pr > 0, 1, 0)
            # flatten
            this_pr_flat = this_pr_bin.sum(1)

            #  Update simultaneous notes counter
            if instrument_name not in self.simultaneous_notes.keys():
                self.simultaneous_notes[instrument_name] = np.zeros((self.num_bins,))

            histo, _ = np.histogram(this_pr_flat, bins=range(self.num_bins+1), range=range(self.num_bins+1))
            self.simultaneous_notes[instrument_name] = self.simultaneous_notes[instrument_name] + histo

        return


if __name__ == '__main__':

    database_path = '/home/leo/Recherche/Databases/Orchestration/arrangement/'
    subsets = [
        # 'bouliane',
        # 'hand_picked_Spotify',
        # 'imslp',
        'liszt_classical_archives'
    ]
    score_iterator = ArrangementIteratorGenerator(
        arrangement_path=database_path,
        subsets=subsets
    )
    savefolder_name = 'liszt_beethov'

    # database_path = '/home/leo/Recherche/Databases/Orchestration/orchestral/'
    # subsets = [
    #     'kunstderfuge'
    # ]
    # score_iterator = OrchestraIteratorGenerator(
    #     folder_path=database_path,
    #     subsets=subsets,
    #     process_file=True
    # )
    # savefolder_name = 'kunst_orchestral'

    simplify_instrumentation_path = 'simplify_instrumentation.json'

    sounding_pitch_boolean = True
    subdivision = 4
    computeStatistics = ComputeStatistics(score_iterator, subdivision, savefolder_name, sounding_pitch_boolean)
    computeStatistics.get_statistics()
