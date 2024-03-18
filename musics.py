import numpy as np

class Staff:
    def __init__(self, id, y_coords) -> None:
        self.id = id
        self.y_coords = y_coords
        self.step_size = float(np.mean(np.diff(y_coords)))
        self.spacer = float(np.mean(np.diff(y_coords))) / 4


class Note:
    def __init__(self, x, y, id=-1, staff_id=0, accidental=0, time=0) -> None:
        self.id = id
        self.x, self.y = x, y
        self.key = "N"
        # Values: 1 (sharp), 0 (none), -1(flat)
        self.accidental = accidental
        self.staff_id = staff_id
        self.time = time

    def debug_info(self):
        print(f"Note ID: {self.id}")
        print(f"Position: ({self.x}, {self.y})")
        print(f"Key: {self.key}")
        print(f"Accidental: {'Sharp' if self.accidental == 1 else 'None' if self.accidental == 0 else 'Flat'}")
        print(f"Staff ID: {self.staff_id}")
        print(f"Time: {self.time}")

    
