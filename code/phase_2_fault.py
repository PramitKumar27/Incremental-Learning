import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# -----------------------------
# 0) Reproducibility (important for experiments)
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 1) Data (same as your baseline)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# -----------------------------
# 2) Model (must match training)
# -----------------------------
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
model.to(device)

checkpoint_path = "./vit_eurosat_clean.pth"  # <-- you create this in training script
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# -----------------------------
# 3) Evaluation (accuracy)
# -----------------------------
@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total

baseline_acc = evaluate_accuracy(model, test_loader)
print(f"Baseline (clean) accuracy: {baseline_acc:.2f}%")

# -----------------------------
# 4) Fault model (bit-flip in float32)
#    Mahdi’s direction: pick random layer, random index, flip one bit
# -----------------------------
def bit_flip_fp32(value_tensor: torch.Tensor) -> torch.Tensor:
    """
    Flip 1 random bit in a float32 representation of the value.
    - We convert the float to a uint32 bit pattern
    - XOR one random bit
    - Convert back to float32
    """
    # Ensure float32 scalar on CPU for bit operations
    v = np.float32(value_tensor.detach().cpu().item())
    bits = v.view(np.uint32)

    bit_to_flip = np.uint32(1 << random.randint(0, 31))
    corrupted_bits = bits ^ bit_to_flip

    corrupted_value = corrupted_bits.view(np.float32)
    return torch.tensor(corrupted_value, device=value_tensor.device, dtype=value_tensor.dtype)

def collect_weight_tensors(model):
    """
    Mahdi script: use only 'weight' tensors.
    This means we inject into ALL layers' weights (not just one block).
    """
    weights = []
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weights.append((name, param))
    return weights

weight_tensors = collect_weight_tensors(model)

# count total number of weight elements (Mahdi’s n)
total_weights = sum(p.numel() for _, p in weight_tensors)
print(f"Total weight elements (n): {total_weights}")

@torch.no_grad()
def inject_faults_in_weights(model, no_faults: int):
    """
    Inject 'no_faults' bit flips:
    - choose random weight tensor
    - choose random element
    - flip 1 bit in float32 representation
    """
    for _ in range(no_faults):
        layer_name, W = random.choice(weight_tensors)
        flat = W.view(-1)
        idx = random.randint(0, flat.numel() - 1)
        flat[idx].copy_(bit_flip_fp32(flat[idx]))

# -----------------------------
# 5) Reliability assessment loop (BER sweep + repetitions)
# -----------------------------
def run_reliability_experiment(bers, repetitions=50):
    results = []

    # Keep a clean copy of parameters in memory (fast restore)
    clean_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ber in bers:
        # Mahdi: no_faults = ber * n (n = number of weight elements)
        no_faults = int(ber * total_weights)

        accs = []
        drops = []

        for r in range(repetitions):
            # restore clean model every run
            model.load_state_dict(clean_state, strict=True)
            model.to(device)
            model.eval()

            if no_faults > 0:
                inject_faults_in_weights(model, no_faults)

            acc = evaluate_accuracy(model, test_loader)
            accs.append(acc)
            drops.append(baseline_acc - acc)

        accs = np.array(accs)
        drops = np.array(drops)

        results.append({
            "ber": ber,
            "no_faults": no_faults,
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std()),
            "mean_drop": float(drops.mean()),
            "worst_acc": float(accs.min()),
        })

        print(f"\nBER={ber:.2e} | no_faults={no_faults}")
        print(f"  mean_acc={accs.mean():.2f}%  std={accs.std():.2f}")
        print(f"  mean_drop={drops.mean():.2f}%  worst_acc={accs.min():.2f}%")

    return results

# Start small (so you can understand results quickly)
bers = [0.0, 1e-7, 5e-7, 1e-6]   # <-- Mahdi-style "fraction of weights corrupted"
repetitions = 50

results = run_reliability_experiment(bers, repetitions=repetitions)
