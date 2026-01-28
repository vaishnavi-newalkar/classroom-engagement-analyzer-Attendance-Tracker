import faiss
import pickle
from insightface.app import FaceAnalysis
import numpy as np

class FaceRecognizer:
    def __init__(
        self,
        index_path="face_identity/faiss.index",
        id_map_path="face_identity/id_map.pkl",
    ):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.index = faiss.read_index(index_path)
        with open(id_map_path, "rb") as f:
            self.id_map = pickle.load(f)



    def detect_and_identify(self, frame):
        faces = self.app.get(frame)
        results = []

        for f in faces:
            emb = f.embedding.astype("float32")
            emb = emb / np.linalg.norm(emb)   # 🔥 REQUIRED
            emb = emb.reshape(1, -1)

            D, I = self.index.search(emb, 1)
            score = float(D[0][0])  # cosine similarity

            if score > 0.35:        # ✅ correct threshold
                sid = self.id_map[I[0][0]]
            else:
                sid = "unknown"

            x1, y1, x2, y2 = map(int, f.bbox)

            results.append({
                "id": sid,
                "bbox": (x1, y1, x2, y2),
                "score": score
            })

        return results
