import glob
import re

import music21


class OrchestraIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    # todo redo
    def __init__(self, folder_path, subsets, process_file):
        self.folder_path = folder_path  # Root of the database
        self.process_file = process_file
        self.subsets = subsets

    def __call__(self, *args, **kwargs):
        it = (
            xml_file
            for xml_file in self.generator()
        )
        return it

    def generator(self):

        folder_paths = []
        for subset in self.subsets:
            folder_paths += glob.glob(f'{self.folder_path}/{subset}/**')

        for folder_path in folder_paths:
            xml_files = glob.glob(folder_path + '/*.xml')
            midi_files = glob.glob(folder_path + '/*.mid')
            if len(xml_files) == 1:
                music_files = xml_files
            elif len(midi_files) == 1:
                music_files = midi_files
            else:
                raise Exception(f"No or too much files in {folder_path}")
            print(music_files)
            # Here parse files and return as a dict containing matrices for piano and orchestra
            if self.process_file:
                try:
                    ret = music21.converter.parse(music_files[0])
                except:
                    with open('dump/shit_files.txt', 'a') as ff:
                        ff.write(music_files[0])

            else:
                ret = music_files[0]

            name = '-'.join(re.split('/', folder_path)[-2:])

            yield {'Piano': None, 'Orchestra': ret, 'name': name}
