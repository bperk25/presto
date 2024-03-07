

class Note:
    def __init__(self, id, x, y, staff_id, accidental) -> None:
        self.id = id
        self.x, self.y = x, y
        # Values: 1 (sharp), 0 (none), -1(flat)
        self.accidental = accidental
        self.staff_id = 0

    def debug_info(self):
        print(f"Note ID: {self.id}")
        print(f"Position: ({self.x}, {self.y})")
        print(f"Note Position: {self.note_pos}")
        print(f"Accidental: {'Sharp' if self.accidental == 1 else 'None' if self.accidental == 0 else 'Flat'}")
        print(f"Octave: {self.octave}")

    # def get_note_name(self, key_signature):     
    #     # Adjust note position based on key signature and accidental
    #     adjusted_note_pos = self.note_pos + key_signature + self.accidental
        
    #     # Handle wrap-around for octave changes
    #     if adjusted_note_pos > 12:
    #         adjusted_note_pos -= 12
    #     elif adjusted_note_pos < 1:
    #         adjusted_note_pos += 12
        
    #     return note_names[adjusted_note_pos]

    
