from midiutil import MIDIFile
from mingus.core import chords


## ---------  MIDI Generation  -------------
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

def note_to_number(note: str, octave: int) -> int:
    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)
    return note


def create_note_num_array(note_objs):
    note_nums = []
    for note in note_objs:
        if note.key == "N": continue
        note_nums.append(note_to_number(note.key, note.octave))

    return note_nums


def create_midi(note_nums, filename = "output.mid", tempo = 120, volume = 100):
    track = 0
    channel = 0
    time = 0  # In beats
    duration = 1  # In beats

    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
    # automatically)
    MyMIDI.addTempo(track, time, tempo)

    for i, pitch in enumerate(note_nums):
        MyMIDI.addNote(track, channel, pitch, time + i, duration, volume)

    with open(filename, "wb") as output_file:
        MyMIDI.writeFile(output_file)