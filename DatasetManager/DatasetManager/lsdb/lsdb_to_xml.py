import os
import re

from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import exclude_list_ids, leadsheet_to_music21, \
    assert_no_time_signature_changes
from DatasetManager.lsdb.lsdb_exceptions import LeadsheetParsingException, \
    LeadsheetTimeSignatureException, LeadsheetKeySignatureException

from bson import ObjectId
from music21.pitch import PitchException


class LsdbConverter:
    """
    Object to handle the creation of local xml databases from LSDB
    """

    # todo other mongodb queries?
    # todo num_elements ...
    def __init__(self,
                 time_signature='4/4',
                 composer=None,
                 songset_ids=None,
                 alternate_name=None
                 ):
        """

        :param time_signature:
        :param composer:
        :param songset_ids: List[str]
        """
        self.time_signature = time_signature
        self.composer = composer
        self.songset_ids = songset_ids
        self.alternate_name = alternate_name
        self.dataset_dir = os.path.join('xml',
                                        self.__repr__())

    def __repr__(self):
        if self.alternate_name is None:
            return f'{self.time_signature.replace("/", "_")}' \
                f'{"_" + self.composer if self.composer else ""}' \
                f'{"_".join(self.songset_ids) if self.songset_ids else ""}'
        else:
            return self.alternate_name

    def make_score_dataset(self):
        """
        Download all LSDB leadsheets, convert them into MusicXML and write them
        in xml folder

        :return:
        """

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        (lsdb_chord_to_notes,
         notes_to_chord_lsdb) = self.compute_lsdb_chord_dicts()

        # todo add query
        with LsdbMongo() as client:
            db = client.get_db()

            # restrict on a specific songset
            query_ids = {'$nin': exclude_list_ids, }
            if self.songset_ids:
                leadsheet_ids_in_songset = []
                for songset_id in self.songset_ids:
                    songset = db.songsets.find_one({
                        '_id': ObjectId(songset_id)
                    })
                    leadsheet_ids_in_songset += [ObjectId(leadsheet_id)
                                                 for leadsheet_id in
                                                 songset['elements']
                                                 ]
                query_ids['$in'] = leadsheet_ids_in_songset

            # query_ids['$in'] = [ObjectId('5193843f58e3383974000eba')]
            query = {'_id': query_ids}
            if self.composer:
                query['composer'] = self.composer

            leadsheets = db.leadsheets.find(
                query,
                no_cursor_timeout=True)
            for leadsheet in leadsheets:
                # discard leadsheet with no title
                if 'title' not in leadsheet:
                    continue
                if os.path.exists(os.path.join(self.dataset_dir,
                                               f'{leadsheet["title"]}.xml'
                                               )):
                    print(leadsheet['title'])
                    print(leadsheet['_id'])
                    print('exists!')
                    continue
                print(leadsheet['title'])
                print(leadsheet['_id'])
                try:
                    self.assert_leadsheet_in_dataset(leadsheet)
                    score = leadsheet_to_music21(leadsheet,
                                                 lsdb_chord_to_notes)
                    export_file_name = os.path.join(
                        self.dataset_dir,
                        f'{self.normalize_leadsheet_name(score.metadata.title)}.xml'
                    )

                    score.write('xml', export_file_name)
                except (LeadsheetKeySignatureException,
                        LeadsheetTimeSignatureException,
                        LeadsheetParsingException,
                        PitchException,
                        ) as e:
                    print(e)
            # close cursor
            leadsheets.close()

    def assert_leadsheet_in_dataset(self, leadsheet):
        if 'time' not in leadsheet:
            raise LeadsheetParsingException(f'Leadsheet '
                                            f'{leadsheet["title"]} '
                                            f'{str(leadsheet["_id"])} '
                                            f'has no "time" field')

        if leadsheet['time'] != self.time_signature:
            raise LeadsheetTimeSignatureException('Leadsheet ' + leadsheet['title'] + ' ' +
                                                  str(leadsheet['_id']) +
                                                  f' is not in {self.time_signature}')
        assert_no_time_signature_changes(leadsheet)

    @staticmethod
    def correct_chord_dicts(chord2notes, notes2chord):
        """
        Modifies chord2notes and notes2chord in place
        to correct errors in LSDB modes (dict of chord symbols with notes)
        :param chord2notes:
        :param notes2chord:
        :return:
        """
        # Add missing chords
        # b5
        notes2chord[('C4', 'E4', 'Gb4')] = notes2chord[('C4', 'E4', 'Gb4')] + ['b5']
        chord2notes['b5'] = ('C4', 'E4', 'Gb4')
        # b9#5
        notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'D#5')] = 'b9#5'
        chord2notes['b9#5'] = ('C4', 'E4', 'G#4', 'Bb4', 'D#5')
        # 7#5#11 is WRONG in the database
        # C4 F4 G#4 B-4 D5 instead of C4 E4 G#4 B-4 D5
        notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'F#5')] = '7#5#11'
        chord2notes['7#5#11'] = ('C4', 'E4', 'G#4', 'Bb4', 'F#5')
        # F#7#9#11 is WRONG in the database
        # M7#4 is WRONG in the database
        chord2notes['M7#4'] = ('C4', 'E4', 'G4', 'B4', 'F#5')
        notes2chord[('C4', 'E4', 'G4', 'B4', 'F#5')] = 'M7#4'
        # 9b5b13 is WRONG in the database
        chord2notes['9b5b13'] = ('C4', 'E4', 'Gb4', 'Bb4', 'D5', 'Ab5')
        notes2chord[('C4', 'E4', 'Gb4', 'Bb4', 'D5', 'Ab5')] = '9b5b13'

    def compute_lsdb_chord_dicts(self):
        # Search LSDB for chord names
        with LsdbMongo() as mongo_client:
            db = mongo_client.get_db()
            modes = db.modes
            cursor_modes = modes.find({})
            chord2notes = {}  # Chord to notes dictionary
            notes2chord = {}  # Notes to chord dictionary
            for chord in cursor_modes:
                notes = []
                # Remove white spaces from notes string
                for note in re.compile("\s*,\s*").split(chord["chordNotes"]):
                    notes.append(note)
                notes = tuple(notes)

                # Enter entries in dictionaries
                chord2notes[chord['mode']] = notes
                if notes in notes2chord:
                    notes2chord[notes] = notes2chord[notes] + [chord["mode"]]
                else:
                    notes2chord[notes] = [chord["mode"]]

            self.correct_chord_dicts(chord2notes, notes2chord)

            return chord2notes, notes2chord

    @staticmethod
    def normalize_leadsheet_name(name):
        return name.replace('/', '-')


def TOM_HEDGES_SONGSET_IDS():
    with LsdbMongo() as client:
        db = client.get_db()
        songsets = db.songsets.find({'authors': 'Tom Hedges'})
        songset_ids = []
        s = 0
        for songset in songsets:
            songset_ids.append(str(songset['_id']))
            s += len(songset['elements'])
    return songset_ids


if __name__ == '__main__':
    # All
    LsdbConverter(songset_ids=TOM_HEDGES_SONGSET_IDS(),
                  alternate_name='Tom').make_score_dataset()

    # Blues
    # LsdbConverter(songset_id='5641fc497cea1f63710ac907').make_score_dataset()

    # Pop
    # LsdbConverter(songset_id='5660684458e3383e7f7b23c6').make_score_dataset()

    # American Songwriter
    # LsdbConverter(songset_id='545a6c8a3004f53efa0382fe').make_score_dataset()

    # Real book & Real book Vol3 2nd Edition
    # LsdbConverter(songset_ids=['545a6c893004f53efa0382e4',
    #                            '545a6c893004f53efa0382e3']
    #               ).make_score_dataset()

    # LsdbConverter(composer='Bill Evans').make_score_dataset()
    # LsdbConverter(composer='Miles Davis').make_score_dataset()
    # LsdbConverter(composer='Duke Ellington').make_score_dataset()
    # LsdbConverter(composer='Fats Waller').make_score_dataset()
    # LsdbConverter(composer='Michel Legrand').make_score_dataset()
    # LsdbConverter(composer='Thelonious Monk').make_score_dataset()
    # LsdbConverter(composer='Charlie Parker').make_score_dataset()
    # LsdbConverter(composer='Antonio Carlos Jobim').make_score_dataset()
    # LsdbConverter(composer='Wayne Shorter').make_score_dataset()
    # LsdbConverter(composer='Sonny Rollins').make_score_dataset()
    # LsdbConverter(composer='John Coltrane').make_score_dataset()
    # LsdbConverter(composer='Chick Corea').make_score_dataset()
    # LsdbConverter(composer='Cole Porter').make_score_dataset()
    # LsdbConverter(composer='Victor Young').make_score_dataset()
    # LsdbConverter(composer='Herbie Hancock').make_score_dataset()
    # LsdbConverter(composer='Pat Metheny').make_score_dataset()
    # LsdbConverter(composer='McCoy Tyner').make_score_dataset()
