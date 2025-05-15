import torch
import torchvision.transforms as T
from PIL import Image
import timm
from torch.nn.functional import normalize
from torch.nn import CosineSimilarity

model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def load_and_embed(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_features(image_tensor)
        if isinstance(embedding, dict):
            embedding = embedding["x_norm_clstoken"]
        embedding = embedding.flatten(1)
    return normalize(embedding, dim=-1)

def compare_images(img1_path, img2_path):
    emb1 = load_and_embed(img1_path)
    emb2 = load_and_embed(img2_path)
    cos = CosineSimilarity(dim=1)
    sim_score = cos(emb1, emb2).item()
    print(f"Cosine similarity between '{img1_path}' and '{img2_path}': {sim_score:.4f}")