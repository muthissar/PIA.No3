import mido
import re

def parse_mido(midi_file):
    mid = mido.MidiFile(midi_file)
    for i, track in enumerate(mid.tracks):
        print(f'Track {i}')
        for message in track:
            if message.type == 'track_name':
                print(message)


if __name__ == '__main__':
    parse_mido('/home/leo/databases/Orchestration/LOP_database_06_09_17/debug/0/symphony_1_1_orch.mid')