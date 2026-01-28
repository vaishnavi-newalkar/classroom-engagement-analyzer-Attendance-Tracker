import csv
from datetime import datetime

class AttendanceTracker:
    def __init__(self, out_file="outputs/attendance.csv"):
        self.marked = set()
        self.out_file = out_file

        with open(self.out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["student_id", "time"])

    def mark(self, student_id: str):
        if student_id == "unknown" or student_id in self.marked:
            return

        self.marked.add(student_id)

        with open(self.out_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, datetime.now().isoformat()])
