
class Note:
    def __init__(self, id, x, y, note_pos, accidental, octave) -> None:
        self.id = id
        self.x, self.y = x, y
        # Vales: 1 -> 12
        self.note_pos = note_pos
        # Values: 1 (sharp), 0 (none), -1(flat)
        self.accidental = accidental
        # Values: 2 -> 6
        self.octave = octave

    def debug_info(self):
        print(f"Note ID: {self.id}")
        print(f"Position: ({self.x}, {self.y})")
        print(f"Note Position: {self.note_pos}")
        print(f"Accidental: {'Sharp' if self.accidental == 1 else 'None' if self.accidental == 0 else 'Flat'}")
        print(f"Octave: {self.octave}")

    def get_note_name(self, key_signature):
        # Dictionary mapping note positions to note names
        note_names = {
            1: "C",
            2: "C#",
            3: "D",
            4: "D#",
            5: "E",
            6: "F",
            7: "F#",
            8: "G",
            9: "G#",
            10: "A",
            11: "A#",
            12: "B"
        }
        
        # Adjust note position based on key signature and accidental
        adjusted_note_pos = self.note_pos + key_signature + self.accidental
        
        # Handle wrap-around for octave changes
        if adjusted_note_pos > 12:
            adjusted_note_pos -= 12
        elif adjusted_note_pos < 1:
            adjusted_note_pos += 12
        
        return note_names[adjusted_note_pos]

    
