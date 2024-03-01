import glob
import os

from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, PAD_SYMBOL
import music21
from music21.chord_symbols.jazz_chords import JazzChord
from DatasetManager.lsdb.lsdb_exceptions import LeadsheetTimeSignatureException, \
    LeadsheetParsingException, LeadsheetKeySignatureException
from bson import ObjectId
import numpy as np

REST = 'R'
NC = 'N.C.'

# dictionary
note_values = {
    'q':  1.,
    'h':  2.,
    'w':  4.,
    '8':  0.5,
    '16': 0.25,
    '32': 0.125,
}

music21_alterations_to_json = {
    '-': 'b',
    '#': '#',
    '':  'n'
}

# list of badly-formatted leadsheets
exclude_list_ids = [
    ObjectId('512dbeca58e3380f1c000000'),  # And on the third day
]


class FakeNote:
    """
    Class used to have SLUR_SYMBOLS with a duration
    """

    def __init__(self, symbol, duration):
        self.symbol = symbol
        self.duration = duration

    def __repr__(self):
        return f'<FakeNote {self.symbol}>'


def general_note(pitch: str, duration: float):
    duration = music21.duration.Duration(duration)

    if pitch == SLUR_SYMBOL:
        return FakeNote(symbol=pitch,
                        duration=duration)
    elif pitch == REST:
        f = music21.note.Rest()
        f.duration = duration
        return f
    else:
        f = music21.note.Note(pitch=pitch)
        f.duration = duration
        return f


def standard_chord_symbol(chord_or_exp_str):
    """

    :param chord_or_exp: string representing a ChordSymbol or a TextExpression
    The text expression is the N.C. symbol
    :return: Corresponding ChordSymbol or TextExpression
    """
    if chord_or_exp_str == NC:
        return music21.expressions.TextExpression(NC)
    elif chord_or_exp_str == START_SYMBOL or chord_or_exp_str == END_SYMBOL:
        print(f'Warning: standard_chord method called with '
              f'{chord_or_exp_str} argument')
        return None
    else:
        return music21.harmony.ChordSymbol(chord_or_exp_str)


def is_tied_left(json_note):
    """

    :param json_note:
    :return: True is the json_note is tied FROM the left
    """
    return ('tie' in json_note
            and
            'stop' in json_note["tie"].split('_'))


def is_tied_right(json_note):
    """

    :param json_note:
    :return: True is the json_note is tied FROM the right
    """
    return ('tie' in json_note
            and
            'start' in json_note["tie"].split('_'))


def note_duration(note_value, dots, time_modification):
    """

    :param time_modification:
    :type note_value: str
    :param note_value: duration of the note regardless of the dots
    :param dots: number of dots (0, 1 or 2)
    :return: the actual duration in beats (float)
    """
    if note_value not in note_values:
        raise LeadsheetParsingException(f'Note value of {note_value}')
    duration = note_values[note_value]
    for dot in range(dots):
        duration *= 1.5
    return duration * time_modification


def getAccidental(json_note):
    """

    :param json_note:
    :return:
    """
    # Pas plus de bémols ni dièses
    if '##' in json_note['keys'][0]:
        return '##'
    if 'bb' in json_note['keys'][0]:
        return '--'
    if 'n' in json_note['keys'][0]:
        return 'becarre'
    if 'b' in json_note['keys'][0]:
        return '-'
    if '#' in json_note['keys'][0]:
        return '#'
    return ''


def getOctave(json_note):
    """

    :param json_note:
    :return: octave as string
    """
    return json_note['keys'][0][-1]


def getUnalteredPitch(json_note):
    """

    :param json_note:
    :return: 'Bb/4' -> B
    """
    return json_note['keys'][0][0]


def retain_altered_pitches_if_tied(altered_pitches, json_note):
    """

    :param altered_pitches: dict
    :param note: json note
    :return:
    """
    pitch = getUnalteredPitch(json_note)
    if pitch in altered_pitches.keys():
        return {pitch: altered_pitches[pitch]}
    else:
        return {}


def altered_pitches_music21_to_dict(alteredPitches):
    """

    :param alteredPitches:
    :return: dictionary {'B': 'b', 'C': ''}
    """
    d = {}
    # todo natural ?
    for pitch in alteredPitches:
        d.update({pitch.name[0]: music21_alterations_to_json[pitch.name[1]]})
    return d


def assert_no_time_signature_changes(leadsheet):
    if 'changes' not in leadsheet or not leadsheet['changes']:
        raise LeadsheetParsingException(f'Leadsheet '
                                        f'{leadsheet["title"]} '
                                        f'{str(leadsheet["_id"])} '
                                        f'has no "changes" field')
    changes = leadsheet['changes']
    for change in changes:
        if ('(timeSig' in change or
                ('timeSignature' in change
                 and
                 not change['timeSignature'] == '')
        ):
            raise LeadsheetTimeSignatureException(f'Leadsheet '
                                                  f'{leadsheet["title"]} '
                                                  f'{str(leadsheet["_id"])} '
                                                  f'has multiple time changes')


def leadsheet_on_ticks(leadsheet, tick_values):
    notes, chords = notes_and_chords(leadsheet)
    eps = 1e-5
    for n in notes:
        i, d = divmod(n.offset, 1)
        flag = False
        for tick_value in tick_values:
            if tick_value - eps < d < tick_value + eps:
                flag = True
        if not flag:
            return False

    return True


def set_metadata(score, lsdb_leadsheet):
    """
    Add metadata extracted from lsdb_leadsheet to score
    In place

    :param score:
    :param lsdb_leadsheet:
    :return:
    """
    score.insert(0, music21.metadata.Metadata())

    if 'title' in lsdb_leadsheet:
        score.metadata.title = lsdb_leadsheet['title']
    if 'composer' in lsdb_leadsheet:
        score.metadata.composer = lsdb_leadsheet['composer']


def notes_and_chords(leadsheet):
    """

    :param leadsheet: music21 score
    :return:
    """
    notes = leadsheet.parts[0].flat.notesAndRests
    notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
    chords = list(leadsheet.parts[0].flat.getElementsByClass(
        [music21.chord_symbols.jazz_chords.JazzChord,
         music21.expressions.TextExpression
         ]))
    return notes, chords


def chord_symbols_from_note_list(all_notes, interval):
    """

    :param all_notes:
    :param interval:
    :return:
    """
    skip_notes = 0
    while True:
        try:
            if skip_notes > 0:
                notes = all_notes[:-skip_notes]
            else:
                notes = all_notes
            chord_relative = music21.chord.Chord(notes)
            chord = chord_relative.transpose(interval)
            chord_root = chord_relative.bass().transpose(interval)
            chord.root(chord_root)
            chord_symbol = music21.harmony.chordSymbolFromChord(chord)
            # print(chord_symbol)
            return chord_symbol
        except (music21.pitch.AccidentalException,
                ValueError) as e:
            # A7b13, m69, 13b9 not handled
            print(e)
            print(chord_relative, chord_relative.root())
            print(chord, chord.root())
            print('========')
            skip_notes += 1


def standard_chord(chord_string):
    assert not chord_string == START_SYMBOL
    assert not chord_string == END_SYMBOL
    if chord_string == NC or chord_string == PAD_SYMBOL:
        return music21.expressions.TextExpression(NC)
    else:
        num_chars = len(chord_string)
        while True:
            try:
                return music21.harmony.ChordSymbol(chord_string[:num_chars])
            except:
                print(f'{chord_string[:num_chars]} not parsable')
                num_chars -= 1
                if num_chars < 0:
                    raise Exception


# def get_root(chord_string):
#     assert not chord_string == START_SYMBOL
#     assert not chord_string == END_SYMBOL
#     assert not chord_string == NC
#     assert not chord_string == PAD_SYMBOL
#
#     if (len(chord_string) > 1
#             and
#             chord_string[1] in ['-', '#']):
#         return chord_string[0:2]
#     else:
#         return chord_string[0]


def chords_duration(bar, number_of_beats):
    """

    :param bar: lsdb bar
    :param number_of_beats:
    :return: list of Durations in beats of each chord in bar
    it is of length num_chords_in_bar + 1
    the first element indicates the duration of a possible
    __ chord (if there are no chords on the first beat)

    Example:(
    if bar has chords on beats 1 and 3
    [d.quarterLength
    for d in self.chords_duration(bar)] = [0, 2, 2]

    if bar has one chord on beat 3
    [d.quarterLength
    for d in self.chords_duration(bar)] = [2, 2]

    if there are no chord (in 4/4):
    [d.quarterLength
    for d in self.chords_duration(bar)] = [4]
    """
    # if there are no chords
    if 'chords' not in bar:
        return [music21.duration.Duration(number_of_beats)]
    json_chords = bar['chords']
    chord_durations = [json_chord['beat']
                       for json_chord in json_chords]
    chord_durations += [number_of_beats + 1]
    chord_durations = np.array(chord_durations, dtype=np.float)

    # beat starts at 1...
    chord_durations -= 1
    chord_durations[1:] -= chord_durations[:-1]

    # convert to music21 objects
    chord_durations = [music21.duration.Duration(d)
                       for d in chord_durations]
    return chord_durations


def chords_in_bar(bar,
                  number_of_beats,
                  lsdb_chord_to_notes):
    """

    :param bar: bar of lsdb_leadsheet
    :param number_of_beats:
    :param lsdb_chord_to_notes: dictionary
    :return: list of music21.chord.Chord with their durations
    if there are no chord on the first beat, a there is a rest
    of the correct duration instead
    """
    chord_durations = chords_duration(bar=bar,
                                      number_of_beats=number_of_beats)
    rest_duration = chord_durations[0]

    # Rest chord during all the measure if no chords in bar
    if 'chords' not in bar:
        rest_chord = music21.note.Rest(duration=rest_duration)
        return [rest_chord]

    json_chords = bar['chords']
    chords = []

    # add Rest chord if there are no chord on the first beat
    if rest_duration.quarterLength > 0:
        rest_chord = music21.note.Rest(duration=rest_duration)
        chords.append(rest_chord)

    for json_chord, duration in zip(json_chords, chord_durations[1:]):
        chord = music21_chord_from_json_chord(json_chord=json_chord,
                                              lsdb_chord_to_notes=lsdb_chord_to_notes)
        chord.duration = duration
        chords.append(chord)

    return chords


def notes_in_bar(bar,
                 altered_pitches_at_key):
    """

    :param bar:
    :param altered_pitches_at_key:
    :return: list of music21.note.Note
    """
    if 'melody' not in bar:
        raise LeadsheetParsingException('No melody')
    bar_melody = bar["melody"]
    current_altered_pitches = altered_pitches_at_key.copy()

    notes = []
    for json_note in bar_melody:
        # pitch is Natural pitch + accidental alteration
        # do not take into account key signatures and previous alterations
        pitch = pitch_from_json_note(
            json_note=json_note,
            current_altered_pitches=current_altered_pitches)

        duration = duration_from_json_note(json_note)

        note = general_note(pitch, duration)
        notes.append(note)
    return notes


def duration_from_json_note(json_note):
    value = (json_note["duration"][:-1]
             if json_note["duration"][-1] == 'r'
             else json_note["duration"])
    dot = (int(json_note["dot"])
           if "dot" in json_note
           else 0)
    time_modification = 1.
    if "time_modification" in json_note:
        # a triolet is denoted as 3/2 in json format
        numerator, denominator = json_note["time_modification"].split('/')
        time_modification = int(denominator) / int(numerator)
    return note_duration(value, dot, time_modification)


def pitch_from_json_note(json_note, current_altered_pitches) -> str:
    """
    Compute the real pitch of a json_note given the current_altered_pitches
    Modifies current_altered_pitches in place!
    :param json_note:
    :param current_altered_pitches:
    :return: string of the pitch or SLUR_SYMBOL if the note is tied
    """
    # if it is a tied note
    if "tie" in json_note:
        if is_tied_left(json_note):
            return SLUR_SYMBOL

    displayed_pitch = (REST
                       if json_note["duration"][-1] == 'r'
                       else json_note["keys"][0])
    # if it is a rest
    if displayed_pitch == REST:
        return REST

    # Otherwise, if it is a true note
    # put real alterations
    unaltered_pitch = getUnalteredPitch(json_note)
    displayed_accidental = getAccidental(json_note)
    octave = getOctave(json_note)
    if displayed_accidental:
        # special case if natural
        if displayed_accidental == 'becarre':
            displayed_accidental = ''
        current_altered_pitches.update(
            {unaltered_pitch: displayed_accidental})
    # compute real pitch
    if unaltered_pitch in current_altered_pitches.keys():
        pitch = (unaltered_pitch +
                 current_altered_pitches[unaltered_pitch] +
                 octave)
    else:
        pitch = unaltered_pitch + octave
    return pitch


def chord_symbols_from_note_list(all_notes, interval):
    """

    :param all_notes:
    :param interval:
    :return:
    """
    # Todo check
    skip_notes = 0
    while True:
        try:
            if skip_notes > 0:
                notes = all_notes[:-skip_notes]
            else:
                notes = all_notes
            chord_relative = music21.chord.Chord(notes)
            chord = chord_relative.transpose(interval)
            chord_root = chord_relative.bass().transpose(interval)
            chord.root(chord_root)
            chord_symbol = music21.harmony.chordSymbolFromChord(chord)
            # print(chord_symbol)
            return chord_symbol
        except (music21.pitch.AccidentalException,
                ValueError) as e:
            # A7b13, m69, 13b9 not handled
            print(e)
            print(chord_relative, chord_relative.root())
            print(chord, chord.root())
            print('========')
            skip_notes += 1


def remove_fake_notes(notes):
    """
    Transforms a list of notes possibly containing FakeNotes
    to a list of music21.note.Note with the correct durations
    :param notes:
    :return:
    """
    previous_note = None

    true_notes = []
    for note in notes:
        if isinstance(note, FakeNote):
            assert note.symbol == SLUR_SYMBOL
            # will raise an error if the first note is a FakeNote
            cumulated_duration += note.duration.quarterLength
        else:
            if previous_note is not None:
                previous_note.duration = music21.duration.Duration(
                    cumulated_duration)
                true_notes.append(previous_note)
            previous_note = note
            cumulated_duration = previous_note.duration.quarterLength

    # add last note
    previous_note.duration = music21.duration.Duration(
        cumulated_duration)
    true_notes.append(previous_note)
    return true_notes


# todo could be merged with remove_fake_notes
def remove_rest_chords(chords):
    """
    Transforms a list of ChordSymbols possibly containing Rests
    to a list of ChordSymbols with the correct durations
    :param chords:
    :return:
    """
    previous_chord = None

    true_chords = []
    for chord in chords:
        if isinstance(chord, music21.note.Rest):
            # if the first chord is a Rest,
            # replace it with a N.C.
            if previous_chord is None:
                previous_chord = music21.expressions.TextExpression(NC)
                cumulated_duration = 0
            cumulated_duration += chord.duration.quarterLength
        else:
            if previous_chord is not None:
                previous_chord.duration = music21.duration.Duration(
                    cumulated_duration)
                true_chords.append(previous_chord)
            previous_chord = chord
            cumulated_duration = previous_chord.duration.quarterLength

    # add last note
    previous_chord.duration = music21.duration.Duration(
        cumulated_duration)
    true_chords.append(previous_chord)
    return true_chords


def music21_chord_from_json_chord(json_chord, lsdb_chord_to_notes):
    """
    Tries to find closest chordSymbol for json_chord using lsdb_chord_to_notes
    :param json_chord:
    :param lsdb_chord_to_notes: dictionnary
    :return:
    """
    assert 'p' in json_chord
    # root
    json_chord_root = json_chord['p']

    # N.C. chords
    if json_chord_root == 'NC':
        return music21.expressions.TextExpression(NC)

    music21_root_pitch = music21.pitch.Pitch(json_chord_root)
    # chord type
    if 'ch' in json_chord:
        json_chord_type = json_chord['ch']
    else:
        json_chord_type = ''

    num_characters_chord_type = len(json_chord_type)
    while True:
        try:
            current_json_chord_type = json_chord_type[:num_characters_chord_type]

            all_notes = lsdb_chord_to_notes[current_json_chord_type]
            all_notes_list = list(all_notes)
            # strip the octave identifier
            all_notes_str_list = [note[:-1] for note in all_notes_list]
            all_notes_str = ' '.join(note for note in all_notes_str_list)

            # all_pitches_list = [music21.pitch.Pitch(note) for note in all_notes_list]
            # all_notes = [note.replace('b', '-')
            #             for note in all_notes]
            chord_symbol = JazzChord(all_notes_str, music21_root_pitch)
            # interval = music21.interval.Interval(
            #    noteStart=music21.note.Note('C4'),
            #    noteEnd=music21.note.Note(json_chord_root))
            # chord_symbol = chord_symbols_from_note_list(
            #    all_notes=all_notes,
            #    interval=interval
            # )
            return chord_symbol
        except (AttributeError, KeyError):
            # if the preceding procedure did not work
            print('Difficult chord')
            print(current_json_chord_type, all_notes_list)
            num_characters_chord_type -= 1


def leadsheet_to_music21(leadsheet, lsdb_chord_to_notes):
    # must convert b to -
    print(leadsheet)
    if 'keySignature' not in leadsheet or not leadsheet['keySignature']:
        raise LeadsheetKeySignatureException(f'Leadsheet {leadsheet["title"]} '
                                             f'has no keySignature')
    key_signature = leadsheet['keySignature'].replace('b', '-')
    key_signature = music21.key.Key(key_signature)
    altered_pitches_at_key = altered_pitches_music21_to_dict(
        key_signature.alteredPitches)

    if 'changes' not in leadsheet:
        raise LeadsheetParsingException('Leadsheet ' + leadsheet['title'] + ' ' +
                                        str(leadsheet['_id']) +
                                        ' do not contain "changes" attribute')

    assert_no_time_signature_changes(leadsheet)
    time_signature = music21.meter.TimeSignature(leadsheet['time'])
    number_of_beats = time_signature.numerator
    assert time_signature.denominator == 4

    chords = []
    notes = []

    score = music21.stream.Score()
    part_notes = music21.stream.Part()
    part_chords = music21.stream.Part()
    for section_index, section in enumerate(leadsheet['changes']):
        for bar_index, bar in enumerate(section['bars']):
            # We consider only 4/4 pieces
            # Chords in bar
            bar_chords = chords_in_bar(bar=bar,
                                       number_of_beats=number_of_beats,
                                       lsdb_chord_to_notes=lsdb_chord_to_notes)
            bar_notes = notes_in_bar(bar,
                                     altered_pitches_at_key)
            chords.extend(bar_chords)
            notes.extend(bar_notes)

    # remove FakeNotes
    notes = remove_fake_notes(notes)
    chords = remove_rest_chords(chords)

    # voice_notes = music21.stream.Voice()
    # voice_chords = music21.stream.Voice()
    part_notes.append(notes)
    part_chords.append(chords)
    for chord in part_chords.flat.getElementsByClass(
            [JazzChord,
             music21.expressions.TextExpression
             ]):
        # put durations to 0.0 as required for a good rendering
        # handles both textExpression (for N.C.) and ChordSymbols
        if isinstance(chord, JazzChord):
            new_chord = chord.__deepcopy__()
            new_chord.duration = music21.duration.Duration(0)
        elif isinstance(chord, music21.expressions.TextExpression):
            new_chord = music21.expressions.TextExpression(NC)
        else:
            raise ValueError
        part_notes.insert(chord.offset, new_chord)
    part_notes = part_notes.makeMeasures(
        inPlace=False,
        refStreamOrTimeRange=[0.0, part_chords.highestTime])

    # add treble clef and key signature
    part_notes.measure(1).clef = music21.clef.TrebleClef()
    part_notes.measure(1).keySignature = key_signature
    score.append(part_notes)
    set_metadata(score, leadsheet)
    # normally we should use this but it does not look good...
    # score = music21.harmony.realizeChordSymbolDurations(score)

    return score


class LeadsheetIteratorGenerator:
    """
    Object that returns a iterator over leadsheet (as music21 scores)
    when called
    :return:
    """

    # todo redo
    def __init__(self, num_elements=None):
        self.num_elements = num_elements

    def __call__(self, *args, **kwargs):
        it = (
            leadsheet
            for leadsheet in self.leadsheet_generator()
        )
        return it

    def leadsheet_generator(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # todo hard coded
        leadsheet_paths = (glob.glob(
            os.path.join(dir_path, 'xml/4_4_all/*.xml')) +
                           glob.glob(
                               os.path.join(dir_path, 'xml/4_4_all/*.mxl'))
                           )
        if self.num_elements is not None:
            leadsheet_paths = leadsheet_paths[:self.num_elements]
        for leadsheet_path in leadsheet_paths:
            try:
                print(leadsheet_path)
                yield music21.converter.parse(leadsheet_path)
            # except (ZeroDivisionError,
            #         KeyError,
            #         UnboundLocalError,
            #         AttributeError,
            #         TypeError) as e:
            except Exception as e:
                print(f'{leadsheet_path} is not parsable')
                print(e)
