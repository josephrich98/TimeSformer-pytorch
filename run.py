import os
import json
from collections import Counter
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from timesformer_pytorch import TimeSformer

data_root = "/mnt/gpussd2/jrich/data/multiphase_ct/USC-and-KITS-segmented"
labels_json = "/mnt/gpussd2/jrich/data/multiphase_ct/code/USC_and_kits_case_labels_A.json"
test_set_cases = ["Keck0064", "Keck0303", "Keck0070", "Keck0391", "Keck0618", "Keck0706", "Keck0210", "Keck0305", "Keck0126", "Keck0548", "Keck0461", "Keck0590", "Keck0537", "Keck0105", "Keck0027", "Keck0314", "Keck0281", "Keck0063", "Keck0505", "Keck0646", "Keck0180", "Keck0001", "Keck0700", "Keck0249", "Keck0731", "Keck0152", "Keck0722", "Keck0410", "Keck0240", "Keck0475", "Keck0685", "Keck0445", "Keck0419", "Keck0725", "Keck0374", "Keck0732", "Keck0503", "Keck0167", "Keck0226", "Keck0543", "Keck0300", "Keck0099", "Keck0279", "Keck0402", "Keck0178", "Keck0469", "Keck0252", "Keck0073", "Keck0341", "Keck0060", "Keck0586", "Keck0550", "Keck0553", "Keck0401", "Keck0579", "Keck0026", "Keck0179", "Keck0118", "Keck0255", "Keck0455", "Keck0691", "Keck0694", "Keck0567", "Keck0539", "Keck0604", "Keck0259", "Keck0272", "Keck0544", "Keck0468", "Keck0144", "Keck0398", "Keck0728", "Keck0393", "Keck0418", "Keck0480", "Keck0357", "Keck0345", "Keck0436", "Keck0181", "Keck0699", "Keck0352", "Keck0187", "Keck0208", "Keck0716", "Keck0628", "Keck0585", "Keck0015", "Keck0657", "Keck0466", "Keck0572", "Keck0214", "Keck0385", "Keck0504", "Keck0306", "Keck0283", "Keck0366", "Keck0707", "Keck0601", "Keck0593", "Keck0714", "Keck0335", "Keck0713", "Keck0023", "Keck0471", "Keck0496", "Keck0474", "Keck0237", "Keck0117", "Keck0400", "Keck0656", "Keck0251", "Keck0569", "Keck0048", "Keck0149", "Keck0603", "Keck0162", "Keck0147", "Keck0723", "Keck0171", "Keck0061", "Keck0005", "Keck0280", "Keck0324", "Keck0282", "Keck0231", "Keck0112", "Keck0013", "Keck0484", "Keck0575", "Keck0501", "Keck0587", "Keck0443", "Keck0384", "Keck0449", "Keck0652", "Keck0028", "Keck0465", "Keck0610", "Keck0378", "Keck0071", "Keck0323", "Keck0704", "Keck0532", "Keck0029", "Keck0212", "Keck0320", "Keck0224"]
batch_size = 8
learning_rate = 1e-4
model_dim = 512
epochs = 100
accurate_time = True
seed = 42

phase_to_expected_time_range = {
    "noncontrast": (0, 5),  # commonly 0 sec
    "corticomedullary": (25, 45),  # commonly 30-35 sec
    "nephrographic": (60, 95),  # commonly 65-80 sec
    "excretory": (270, 700)  # commonly 300-375 sec
}
phase_name_synonyoms = {
    "noncontrast": ["noncontrast", "precontrast", "unenhanced", "native"],
    "corticomedullary": ["corticomedullary", "arterial", "cortico-medullary", "cortico medullary"],
    "nephrographic": ["nephrographic", "venous", "nephro-graphic", "nephro graphic", "nephrogenic"],
    "excretory": ["excretory", "delayed", "excretory phase", "delayed phase"]
}
for standard_name, synonyms in phase_name_synonyoms.items():
    for syn in synonyms:
        phase_to_expected_time_range[syn] = phase_to_expected_time_range[standard_name]

def set_seed(seed: int = 42):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_sequence_from_folder(
    folder,
    phase_to_expected_time_range,
    max_frames=4,
    target_shape=(256, 256),
    dtype=np.float32,
):
    """
    Returns:
        video: (T, 1, H, W)
        mask:  (T,)  True = phase present, False = missing
    """

    # canonical phase order by expected start time
    ordered_phases = sorted(
        phase_to_expected_time_range.keys(),
        key=lambda k: phase_to_expected_time_range[k][0]
    )[:max_frames]

    frames = []
    mask = []
    times = []

    for phase in ordered_phases:
        path = os.path.join(folder, f"{phase}.npy")

        beginning_time = phase_to_expected_time_range[phase][0]
        times.append(beginning_time)

        if os.path.exists(path):
            arr = np.load(path).astype(dtype)

            # ensure shape (1, H, W)
            if arr.ndim == 2:
                arr = arr[None, :, :]
            elif arr.ndim == 3 and arr.shape[0] == 1:
                pass
            else:
                raise ValueError(f"{phase}.npy has invalid shape {arr.shape}")

            # validate spatial dimensions
            if arr.shape[1:] != target_shape:
                raise ValueError(
                    f"{phase}.npy has shape {arr.shape[1:]}, expected {target_shape}"
                )

            frames.append(arr)
            mask.append(True)

        else:
            # missing phase → pad but mask it out
            frames.append(np.zeros((1, *target_shape), dtype=dtype))
            mask.append(False)

    video = np.stack(frames, axis=0)   # (T, 1, H, W)
    mask  = np.array(mask, dtype=bool) # (T,)

    return torch.from_numpy(video), torch.from_numpy(mask), torch.tensor(times, dtype=torch.float32)

set_seed(seed)

with open(labels_json) as f:
    labels = json.load(f)  # a dict from sample folder name to int label

train_filenames = [
    f for f in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, f)) and f in labels and f not in test_set_cases
]

test_filenames = [
    f for f in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, f)) and f in labels and f in test_set_cases
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeSformer(
    dim = model_dim,
    image_size = 256,
    patch_size = 16,
    num_frames = 4,
    num_classes = 3,
    depth = 12,
    heads = 8,
    dim_head =  64,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_labels = {k: v for k, v in labels.items() if k not in test_set_cases}
test_labels = {k: v for k, v in labels.items() if k in test_set_cases}
train_counts = Counter(train_labels.values())
total_counts = sum(train_counts.values())
num_classes = len(train_counts)
weights = torch.tensor([total_counts/train_counts[c] for c in range(num_classes)], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

print("Starting training...")
for epoch in range(epochs):
    #* Train
    random.shuffle(train_filenames)  # in-place randomization
    
    model.train()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []
    for i in range(0, len(train_filenames), batch_size):
        batch = train_filenames[i:i + batch_size]

        # batch is a list of folder names
        videos, masks, times, batch_labels = [], [], [], []
        for case_name in batch:
            folder_path = os.path.join(data_root, case_name)
            video, mask, time = load_sequence_from_folder(
                folder_path,
                phase_to_expected_time_range=phase_to_expected_time_range,
                max_frames=4,
                target_shape=(256, 256),
                dtype=np.float32
            )
            videos.append(video)
            masks.append(mask)
            times.append(time)

            if case_name not in labels:
                raise ValueError(f"Case {case_name} not found in labels JSON")
            label = train_labels[case_name]
            batch_labels.append(label)

        videos = torch.stack(videos, dim=0)  # (B, T=4, C=1, H=256, W=256)
        masks = torch.stack(masks, dim=0)    # (B, T=4)
        times = torch.stack(times, dim=0)    # (B, T=4)

        videos = videos.repeat(1, 1, 3, 1, 1)  # (B, T=4, C=3, H=256, W=256) - make 3 channels

        videos = videos.to(device)
        masks = masks.to(device)
        times = times.to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        if not accurate_time:
            times = None  # disable accurate time input

        optimizer.zero_grad()
        pred = model(videos, mask=masks, times=times)
        loss = criterion(pred, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        y_true.extend(batch_labels.cpu().numpy().tolist())
        y_pred_batch = pred.argmax(dim=1)
        y_pred.extend(y_pred_batch.cpu().numpy().tolist())
        y_probs_batch = torch.softmax(pred, dim=1)[:, 1]  #
        y_probs.extend(y_probs_batch.detach().cpu().numpy().tolist())

    avg_train_loss = total_loss / (len(train_filenames) // batch_size + 1)
    train_accuracy = accuracy_score(y_true, y_pred)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

    #* Validation
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []
    for filename in test_filenames:
        folder_path = os.path.join(data_root, filename)
        video, mask, times = load_sequence_from_folder(
            folder_path,
            phase_to_expected_time_range=phase_to_expected_time_range,
            max_frames=4,
            target_shape=(256, 256),
            dtype=np.float32
        )
        label = test_labels[filename]
        video = video.unsqueeze(0)  # (1, T=4, C=1, H=256, W=256)
        mask = mask.unsqueeze(0)      # (1, T=4)
        times = times.unsqueeze(0)    # (1, T=4)

        video = video.repeat(1, 1, 3, 1, 1)  # (1, T=4, C=3, H=256, W=256) - make 3 channels

        video = video.to(device)
        mask = mask.to(device)
        times = times.to(device)
        label = torch.tensor([label], dtype=torch.long).to(device)

        if not accurate_time:
            times = None  # disable accurate time input
        
        with torch.no_grad():
            pred = model(video, mask=mask, times=times)  # (1, num_classes=3)

        loss = criterion(pred, label)
        total_loss += loss.item()
        y_true.append(label.item())
        y_pred_batch = pred.argmax(dim=1)
        y_pred.append(y_pred_batch.cpu().numpy().item())
        y_probs_batch = torch.softmax(pred, dim=1)[:, 1]  #
        y_probs.append(y_probs_batch.detach().cpu().numpy().item())
    avg_val_loss = total_loss / len(test_filenames)
    val_accuracy = accuracy_score(y_true, y_pred)
    print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
