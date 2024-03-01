# Keys must match values of instrument_grouping
# 0 means instrument is not used in the db
def get_instrumentation():
    ordered_instruments = {
        "Piccolo": 1,
        "Flute": 2,
        "Oboe": 2,
        "Clarinet": 2,
        "Bassoon": 2,
        "Horn": 3,
        "Piano": 15,
        "Trumpet": 2,
        "Trombone": 2,
        "Tuba": 1,
        "Violin_1": 3,
        "Violin_2": 3,
        "Viola": 2,
        "Violoncello": 2,
        "Contrabass": 2,
        "Remove": 0
    }
    return ordered_instruments


def get_instrumentation_grouped():
    ordered_instruments = {
        "Woodwind": 9,
        "String": 12,
        "Brass": 8,
        "Piano": 9,
        "Remove": 0
    }
    return ordered_instruments
