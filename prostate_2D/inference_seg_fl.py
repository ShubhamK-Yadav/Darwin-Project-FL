import os
import torch
import numpy as np
import SimpleITK as sitk
from custom.utils.training_utils import SemanticSeg  # Same as used in FL
from tqdm import tqdm

# === CONFIGURE THESE PATHS ===
MODEL_PATH = "/users/aca21sky/prostate/prostate_2D/job_configs/picai_fedsemi/workspace_picai_fedsemi/server/simulate_job/app_server/FL_global_model.pt"  # Final FL-trained model checkpoint
TEST_DIR = "/users/aca21sky/prostate/preprocessed_output/nnUNet_test_data"   # Contains subject_0000.nii.gz, subject_0001.nii.gz, ...
OUTPUT_DIR = "/users/aca21sky/prostate/segmentation_result"     # Where predicted masks will go

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model structure ===
model = SemanticSeg(
    lr=1e-4,
    n_epoch=1,
    channels=3,
    num_classes=2,
    input_shape=(384, 384),
    batch_size=1,
    num_workers=2,
    device="0",
    pre_trained=False,
    ckpt_point=False,
    use_fp16=False,
    transformer_depth=18,
    use_transfer_learning=True
).net

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
print(checkpoint.keys())
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(DEVICE)

print("Model loaded. Starting inference...")

# === Perform inference ===
for file in tqdm(sorted(os.listdir(TEST_DIR))):
    if not file.endswith("_0000.nii.gz"):
        continue

    subject_id = file.replace("_0000.nii.gz", "")

    # Load 3 modalities
    paths = [os.path.join(TEST_DIR, f"{subject_id}_{i:04d}.nii.gz") for i in range(3)]
    if not all(os.path.exists(p) for p in paths):
        print(f"[WARN] Missing modalities for {subject_id}, skipping.")
        continue

    # Read and stack images
    imgs = [sitk.ReadImage(p) for p in paths]
    arrays = [sitk.GetArrayFromImage(im) for im in imgs]
    image_np = np.stack(arrays)  # Shape: [3, D, H, W]

    # Run slice-by-slice inference (2D model)
    pred_mask = np.zeros(image_np.shape[1:], dtype=np.uint8)  # [D, H, W]
    for i in range(image_np.shape[1]):  # Loop over depth
        input_slice = image_np[:, i, :, :]  # [3, H, W]
        input_tensor = torch.tensor(input_slice.astype(np.float32)).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]

            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

        pred_mask[i] = pred

    # === NEW: Save as .npy file instead of .nii.gz ===
    output_path = os.path.join(OUTPUT_DIR, f"{subject_id}.npy")
    np.save(output_path, pred_mask)
    print(f"[INFO] Saved prediction: {output_path}")

