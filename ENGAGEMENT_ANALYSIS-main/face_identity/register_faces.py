import os
import cv2
import faiss
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# =========================
# PATHS (MATCH FaceRecognizer)
# =========================
DATASET_DIR = "face_identity/students"
INDEX_PATH = "face_identity/faiss.index"
MAP_PATH = "face_identity/id_map.pkl"

os.makedirs("face_identity", exist_ok=True)

# =========================
# InsightFace
# =========================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
id_map = []

print("[INFO] Registering faces...")

for person_id in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_id)

    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"[WARN] No face in {img_path}")
            continue

        # Take largest face
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        emb = face.embedding.astype("float32")

        # 🔥 L2 normalize (VERY IMPORTANT)
        emb /= np.linalg.norm(emb)

        embeddings.append(emb)
        id_map.append(person_id)

        print(f"[OK] {person_id} ← {img_name}")

# =========================
# FAISS
# =========================
embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
with open(MAP_PATH, "wb") as f:
    pickle.dump(id_map, f)

print("✅ Face registration COMPLETE")
print("Total embeddings:", len(id_map))
