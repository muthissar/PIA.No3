import glob
import csv
import os


class WavetableIteratorGenerator:
    """
    Object that returns a iterator over midi files when called
    :return:
    """
    def __init__(self, num_elements=None):
        self.path = f'{os.path.expanduser("~")}/Data/databases/Wavetables'
        self.num_elements = num_elements

    def __call__(self, *args, **kwargs):
        it = (xml_file for xml_file in self.generator())
        return it

    def __str__(self) -> str:
        ret = 'WavetableIterator'
        if self.num_elements is not None:
            ret += f'_{self.num_elements}'
        return ret

    def generator(self):
        wt_files = (glob.glob(os.path.join(self.path, '**', '*.wav'), recursive=True))

        if self.num_elements is not None:
            wt_files = wt_files[:self.num_elements]

        split_csv_path = os.path.join(self.path, 'split.csv')
        if not os.path.exists(split_csv_path):
            self._create_split_csv(wt_files, split_csv_path)
        with open(split_csv_path, 'r') as csv_file:
            split_csv = csv.DictReader(csv_file, delimiter='\t')
            # create dict so that we can close the file
            d = {}
            for row in split_csv:
                wt_file = row['wt_filename']
                split = row['split']
                d[wt_file] = split
        for wt_file, split in d.items():
            print(wt_file)
            yield wt_file, split

    def _create_split_csv(self, wt_files, split_csv_path):
        print('Creating CSV split')
        with open(split_csv_path, 'w') as file:
            # header
            header = 'wt_filename\tsplit\n'
            file.write(header)
            for k, wt_file_path in enumerate(wt_files):
                # 90/10/0 split
                if k % 10 == 0:
                    split = 'validation'
                else:
                    split = 'train'
                entry = f'{wt_file_path}\t{split}\n'
                file.write(entry)