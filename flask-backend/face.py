import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.nn.functional import normalize
from torch.nn import CosineSimilarity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize face detector and embedder
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def detect_face(image_path):
    """
    Detects a face in the given image using MTCNN.
    Returns aligned face tensor if successful, else None.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        print(f"[ERROR] Cannot open image: '{image_path}' (Unidentified or corrupt)")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load image '{image_path}': {e}")
        return None

    try:
        face = mtcnn(img)
        if face is None:
            print(f"[WARN] No face detected in '{image_path}'")
        return face
    except Exception as e:
        print(f"[ERROR] Face detection failed for '{image_path}': {e}")
        return None

def embed_face(image_path):
    """
    Extracts the face embedding from the image.
    Returns normalized 512-dim embedding if face is found, else None.
    """
    face_tensor = detect_face(image_path)
    if face_tensor is None:
        return None

    try:
        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 160, 160]
            embedding = model(face_tensor)
            return normalize(embedding, dim=-1)  # Shape: [1, 512]
    except Exception as e:
        print(f"[ERROR] Embedding failed for '{image_path}': {e}")
        return None

def compare_images(img1_path, img2_path):
    """
    Compares two face images using cosine similarity between embeddings.
    """
    emb1 = embed_face(img1_path)
    emb2 = embed_face(img2_path)

    if emb1 is None or emb2 is None:
        print("[WARN] Face not detected in one or both images.")
        return

    try:
        cos = CosineSimilarity(dim=1)
        sim_score = cos(emb1, emb2).item()
        print(f"Cosine similarity between '{img1_path}' and '{img2_path}': {sim_score:.4f}", flush=True)
    except Exception as e:
        print(f"[ERROR] Similarity computation failed: {e}")
