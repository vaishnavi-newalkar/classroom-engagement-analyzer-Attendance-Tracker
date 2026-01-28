class StudentMetrics:
    def __init__(self):
        self.data = {}

    def update(self, student_id: str, label: str, confidence: float):
        if student_id == "unknown":
            return

        if student_id not in self.data:
            self.data[student_id] = {
                "frames": 0,
                "active_frames": 0
            }

        self.data[student_id]["frames"] += 1

        if label in {"ATTENTIVE", "FOCUSED"}:
            self.data[student_id]["active_frames"] += 1

    def summary(self):
        summary = {}
        for sid, v in self.data.items():
            summary[sid] = round(
                v["active_frames"] / max(v["frames"], 1), 2
            )
        return summary
