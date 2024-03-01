import copy
import xml.etree.cElementTree as ET
import music21
from music21.chord import Chord
from music21.chord_symbols.chord_maps import voicing_to_chord_id, \
    chord_id_to_chord_dict, \
    chord_name_to_chord_id
from music21.expressions import TextExpression


class JazzChord(TextExpression):
    def __init__(self, identifier=None, root_pitch=None, bass_pitch=None):
        """
        Initializes a Jazz Chord using either a string 
        or a list of pitches

        :param identifier: string or list of music21 pitches
        :param root_pitch: music21.pitch object, must be set as None 
                            if identifier string is not in C
        :param bass_pitch: music21.pitch object, set as None if no 
                            additonal bass note 
        """
        super(JazzChord, self).__init__(content=None)
        self._root = None # music21.pitch object
        self._bass = bass_pitch # TODO: add this in the init functions
        self._chord = None
        self._chord_name = None
        self._xml_name = None
        self._voicing_in_c = None
        self._xml_string = None
        self._degrees = None  # list
        self.is_initialized = False
        self._content = None
        # intialize from string
        if isinstance(identifier, str):
            self.init_chord_from_string(identifier, root_pitch)
        # or intialize from pitch list
        elif isinstance(identifier, list):
            self.init_chord_from_pitch_list(identifier, root_pitch)
        # or initialize from music21 Chord
        elif isinstance(identifier, Chord):
            self.init_chord_from_pitch_list(identifier.pitches, root_pitch)
        # or initialize from chord id
        elif isinstance(identifier, int):
            self.init_chord_type_from_id(identifier, root_pitch=root_pitch)
        else:
            self.is_initialized = False
            # print("Input must be a string or list")
        # TODO: update field content for a proper print

    def init_chord_from_string(self, voicing_string, root_pitch=None):
        """
        Converts a chord-string to Jazz Chord Representation

        :param chord_string: string, must contain only 
                             pitch letters, accidentals, 
                             separated by spaces
        :param root_pitch: music21.pitch object
        """
        # get music21 pitch list from string
        pitch_list = self.get_pitchlist_from_string(voicing_string)
        # initialize using pitch list
        self.init_chord_from_pitch_list(pitch_list, root_pitch)

    def init_chord_from_pitch_list(self, pitch_list, root_pitch):
        """
        Converts a list of pitches to Jazz Chord Representation

        :param pitch_list: list of music21 pitches
        :param root_pitch: music21.pitch object
        """
        # transpose pitch list to C
        root = pitch_list[0]
        c = music21.pitch.Pitch('C')
        transposition_interval = music21.interval.Interval(root, c)
        transposed_pitch_list = [
            p.transpose(transposition_interval)
            for p in pitch_list
        ]
        # initialize default major chord if pitch_list has only 1 entry
        if len(transposed_pitch_list) == 1:
            self.id = 1  # id for default major chord
            print('Initializing default chord')
        else:
            # create voicing string in C
            voicing_str = ' '.join([p.name for p in transposed_pitch_list])
            voicing_str = voicing_str.replace('-', 'b')
            try:
                self.id = voicing_to_chord_id[voicing_str]
            except KeyError:
                raise KeyError(
                    "Voicing string '" + voicing_str +
                    "' not available in dictionary")
        self.init_chord_type_from_id(self.id, root_pitch)

    def init_chord_type_from_id(self, chord_id, root_pitch):
        '''
        Initializes chord from chord-id and root

        :param chord_id: int, 
        :param root_pitch: music21.pitch object
        '''
        # update id:
        self.id = chord_id
        # set root
        if root_pitch is None:
            self._root = music21.pitch.Pitch('C')
        else:
            self._root = root_pitch
        # update all other chord parameters
        self._chord = chord_id_to_chord_dict[self.id]
        self._chord_name = self._chord['name']
        self._xml_name = self._chord['xml_name']
        self._voicing_in_c = self._chord['voicing']
        self._xml_string = self._chord['xml_string']
        self._degrees = self._chord['degrees']
        self.is_initialized = True
        self.set_content()

    def set_content(self):
        """
        Sets the content field for use by TextExpression property
        """
        if self._root.alter == 1:
            alter = '#'
        elif self._root.alter == -1:
            alter = 'b'
        else:
            alter = ''
        self._content = self._root.step + alter + ',' + self._chord_name

    def get_pitchlist_from_string(self, voicing_string):
        """
        Checks if the chord string has valid characters
        Creates a list of music21 notes
        :param voicing_string: string,
        """
        pitch_list = []
        # replace 'b' accidental with '-' for music21
        voicing_string = voicing_string.replace('b', '-')
        # split into pitch strings
        pitch_strings = voicing_string.split()
        # chords cannot have less than 2 pitchess
        # TODO: only checks the number of pitches now. Need to
        #       check the characters later
        if not pitch_strings:
            raise ValueError("Invalid chord string")
        else:
            # initialize octave
            octave = 4
            # iterate through pitches
            # string must contain pitches in sequence for now
            for pitch_idx, pitch_string in enumerate(pitch_strings):
                # adjust octave
                if pitch_idx == 0:
                    pitch_string += str(octave)
                    pitch_list.append(music21.pitch.Pitch(pitch_string))
                else:
                    # get midi of current pitch
                    curr_pitch = music21.pitch.Pitch(pitch_string)
                    curr_midi = curr_pitch.midi
                    # get midi of previous pitch
                    prev_midi = pitch_list[pitch_idx - 1].midi
                    # adjust octave if needed
                    if curr_midi < prev_midi:
                        octave += 1
                    pitch_string += str(octave)
                    pitch_list.append(music21.pitch.Pitch(pitch_string))
                pitch_string += str(octave)
            return pitch_list

    def to_xml(self):
        """
        Converts chord to xml 
        Returns ET Element object
        """
        # check if chord is initialized
        if not self.is_initialized:
            raise ValueError("Chord is not Initialized")
        # create harmony tag
        harmony = ET.Element("harmony", )
        harmony.set('print-frame', 'no')
        # add sub-elements
        self.add_xml_element(harmony, "root")
        self.add_xml_element(harmony, "kind")
        self.add_xml_element(harmony, "bass")
        self.add_xml_element(harmony, "degree")
        return harmony

    def add_xml_element(self, harmony, element_type):
        """
        Adds a specific sub-element type to the harmony element 

        :param harmony: ET Element object
        :param element_type: string, harmony subelement
                             ('root', 'kind', 'degree' or 'bass')
        """
        if element_type == "root":
            self.add_root_xml_element(harmony)
        elif element_type == "kind":
            self.add_kind_xml_element(harmony)
        elif element_type == "degree":
            self.add_degree_xml_element(harmony)
        elif element_type == "bass":
            self.add_bass_xml_element(harmony)
        else:
            raise ValueError("Invalid sub-element type for musicXML harmony")

    def add_root_xml_element(self, harmony):
        """
        Adds the root sub-element type to the harmony element 

        :param harmony: ET Element object
        """
        root = ET.SubElement(harmony, "root")
        ET.SubElement(root, "root-step").text = self._root.step
        root_alter = self._root.alter
        if root_alter == 1:
            ET.SubElement(root, "root-alter").text = '+1'
        elif root_alter == -1:
            ET.SubElement(root, "root-alter").text = '-1'

    def add_kind_xml_element(self, harmony):
        """
        Adds the kind sub-element type to the harmony element 

        :param harmony: ET Element object
        """
        kind = ET.SubElement(harmony, "kind")
        if self._chord_name != "":
            kind.set('text', self._chord_name)
        kind.text = self._xml_name

    def add_degree_xml_element(self, harmony):
        """
        Adds the degree sub-element type to the harmony element 

        :param harmony: ET Element object
        """
        if len(self._degrees) > 0:
            for degree_str in self._degrees:
                self.add_degree_element(harmony, degree_str)

    def add_bass_xml_element(self, harmony):
        """
        Adds the bass sub-element type to the harmony element 

        :param harmony: ET Element object
        """
        if self._bass is not None:
            bass = ET.SubElement(harmony, "bass")
            ET.SubElement(bass, "bass-step").text = self._bass.step
            bass_alter = self._bass.alter
            if bass_alter == 1:
                ET.SubElement(bass, "bass-alter").text = '+1'
            elif bass_alter == -1:
                ET.SubElement(bass, "bass-alter").text = '-1'

    def add_degree_element(self, harmony, degree_str):
        """
        Creates a degree SubElement in the harmony xml tree

        :param harmony: ET Element object
        :param degree_str: string, specifying chord degree type
        """
        # check degree string
        assert (len(degree_str) > 3)

        # create the sub element
        degree = ET.SubElement(harmony, "degree")

        # degree type
        degree_type = degree_str[0:3]
        assert (len(degree_type) == 3)
        if degree_type == 'sub':
            degree_value = degree_str[3:]
            degree_alter = '0'
            degree_type = 'subtract'
        elif degree_type == 'add' or degree_type == 'alt':
            if degree_type == 'alt':
                degree_type = 'alter'
            if degree_str[3] == 'b':
                degree_alter = '-1'
                degree_value = degree_str[4:]
            elif degree_str[3] == '#':
                degree_alter = '+1'
                degree_value = degree_str[4:]
            else:
                degree_alter = '0'
                assert (degree_str[3:].isdigit())
                degree_value = degree_str[3:]
        else:
            raise ValueError("Invalid degree type")
            # create subelements of degree
        ET.SubElement(degree, "degree-value").text = degree_value
        ET.SubElement(degree, "degree-alter").text = degree_alter
        ET.SubElement(degree, "degree-type").text = degree_type

    def get_pitchlist_from_chord(self):
        '''
        Returns the pitches based on the voicing string and root of the 
        chord
        '''
        assert (self.is_initialized)
        root = self._root
        voicing_in_c = self._voicing_in_c
        pitchlist_in_c = self.get_pitchlist_from_string(voicing_in_c)

        c = music21.pitch.Pitch('C')
        transposition_interval = music21.interval.Interval(c, root)
        transposed_pitch_list = [
            p.transpose(transposition_interval)
            for p in pitchlist_in_c
        ]
        return transposed_pitch_list

    def transpose(self, value, *, inPlace=False):
        if not inPlace:
            post = copy.deepcopy(self)
        else:
            post = self

        transposed_root = post._root.transpose(value)
        post.init_chord_type_from_id(chord_id=post.id,
                                     root_pitch=transposed_root)

        if not inPlace:
            # what does that mean?
            # found in Note music21 class
            # post.derivation.method = 'transpose'
            return post
        else:
            return None

    @staticmethod
    def get_chord_id_from_name_str(name_str):
        '''
        Returns the chord-id from name string

        :param name_str: string,
        '''
        chord_id = chord_name_to_chord_id[name_str]
        return int(chord_id)

    @staticmethod 
    def get_chord_id_from_abc(abc_chord_str):
        '''
        Returns the chord_id from abc chord string

        :param abc_chord_str: string, 
        '''
        def parse_abc_chord_string(abc_chord_str):
            '''
            Parses the abc_chord_string and returns the 
            root, bass and chord_type strings

            :param abc_chord_str: string,
            '''
             # check for bass notes
            if '/' in abc_chord_str:
                chord_elements = abc_chord_str.split('/')
                # should have exactly 2 elements
                if len(chord_elements) != 2:
                    raise ValueError('Improper formatting for abc chord')
                else:
                    chord_str = chord_elements[0]
                    bass_str = chord_elements[1]
            else:
                bass_str = None
                chord_str = abc_chord_str
            root_step = chord_str[0]
            if chord_str[1] == '#' or chord_str[1] == 'b':
                root_str = root_step + chord_str[1]
                if len(chord_str) > 2:
                    chord_type = chord_str[2:]
                else:
                    chord_type = None
            else:
                root_str = root_step
                if len(chord_str) > 1:
                    chord_type = chord_str[1:]
                else:
                    chord_type = None
            root_str = root_str.replace('b', '-')
            if bass_str is not None:
                bass_str = bass_str.replace('b', '-')
            return root_str, bass_str, chord_type

        def get_chord_id_from_typestr(chord_typestr):
            '''
            Returns the jazz chord id from chord type string
            
            :param chord_typestr, string
            '''
            # TODO: write this method properly
            try:
                chord_id = chord_name_to_chord_id[chord_typestr]
            except KeyError:
                # return default major chord
                raise ValueError('Invalid chord type string: ', chord_typestr)
            return chord_id

        # get the root, bass and chord type strings
        root_str, bass_str, chord_type = parse_abc_chord_string(abc_chord_str)
        # try converting root and bass strings to music21 pitches
        try:
            root_pitch = music21.pitch.Pitch(root_str)
        except ValueError:
            raise ValueError('Invalid root in abc chord')
        if bass_str is not None:
            try: 
                bass_pitch = music21.pitch.Pitch(bass_str)
            except ValueError:
                raise ValueError('Invalid bass in abc chord')
        else:
            bass_pitch = None
        if chord_type is None:
            chord_id = 1
        else:
            chord_id = get_chord_id_from_typestr(chord_type)
        return root_pitch, bass_pitch, chord_id

    @property
    def root(self):
        root = self._root
        return root

    @property
    def chord_name(self):
        return self._chord_name

    @property
    def xml_name(self):
        return self._xml_name

    @property
    def content(self):
        return self._content
