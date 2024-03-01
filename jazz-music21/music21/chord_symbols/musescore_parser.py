import xml.etree.ElementTree as ET


def parse_xml_chord(xml_chord: ET.Element):
    ET.dump(xml_chord)
    voicing = xml_chord.find('voicing').text

    name = xml_chord.find('name').text
    # pb with major chord
    if name is None:
        name = ''

    xml_name = xml_chord.find('xml').text
    degrees = [
        degree.text for degree in
        xml_chord.iter('degree')
    ]
    id = xml_chord.attrib['id']
    chord = {
        'id':         id,
        'name':       name,
        'xml_name':   xml_name,
        'voicing':    voicing,
        'xml_string': ET.tostring(xml_chord),
        'degrees':    degrees
    }
    return chord


def generate_chord_tables():
    with open('chord_maps.py', 'w') as f:
        tree = ET.parse('chords.xml')
        root = tree.getroot()

        # generate id -> chord dict
        f.write('chord_id_to_chord_dict = {\n')
        for xml_chord in root.iter('chord'):
            chord_dict = parse_xml_chord(xml_chord)
            f.write(f"{chord_dict['id']}:"
                    f"{chord_dict},\n")
        f.write('}\n\n')

        # generate voicing -> id
        f.write('voicing_to_chord_id = {\n')
        for xml_chord in root.iter('chord'):
            chord_dict = parse_xml_chord(xml_chord)
            f.write(f"'{chord_dict['voicing']}':"
                    f"{chord_dict['id']},\n")
        f.write('}\n\n')

        # generate chord_name -> id
        f.write('chord_name_to_chord_id = {\n')
        for xml_chord in root.iter('chord'):
            chord_dict = parse_xml_chord(xml_chord)
            f.write(f"'{chord_dict['name']}':"
                    f"{chord_dict['id']},\n")
        f.write('}')


if __name__ == '__main__':
    generate_chord_tables()
