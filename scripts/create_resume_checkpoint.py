"""
Creates a generator checkpoint for resuming sample_unlearning_cls_distr.py
from a partial run, by replaying the generator advances without running inference.

Assumptions (verified from the run that crashed):
  - seed = 188
  - 4 GPUs -> 40 class_theme_pairs split evenly -> batch_size = 10 per process
  - SD1.5, 512x512 images -> latent shape (10, 4, 64, 64)
  - Completed: Architectures, Bears, Birds, Butterfly (50 themes each)
               + Cats: Abstractionism, Artist_Sketch, Blossom_Season, Bricks, Byzantine
  - Crashed mid-save on Cats/Cartoon -> re-run that iteration on resume
"""

import torch
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from UnlearnCanvas_resources.const import class_available, theme_available

OUTPUT_DIR = "/leonardo_scratch/large/userexternal/ecassano/saemnesia_repo_results"
SEED = 188
NUM_PROCESSES = 4
IMAGE_SIZE = 512
LATENT_SIZE = IMAGE_SIZE // 8  # 64
CHANNELS = 4

theme_avail = [t for t in theme_available if t != "Seed_Images"]
pairs_per_process = (2 * len(class_available)) // NUM_PROCESSES  # 40 / 4 = 10
latent_shape = (pairs_per_process, CHANNELS, LATENT_SIZE, LATENT_SIZE)

# Count completed iterations
completed_classes = ["Architectures", "Bears", "Birds", "Butterfly"]
completed_cats_themes = ["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine"]

n_calls = len(completed_classes) * len(theme_avail) + len(completed_cats_themes)
last_completed = ("Cats", "Byzantine")

print(f"Advancing generator {n_calls} times with shape {latent_shape}...")

generator = torch.Generator(device="cpu").manual_seed(SEED)
for i in range(n_calls):
    torch.randn(latent_shape, generator=generator)

checkpoint_path = os.path.join(OUTPUT_DIR, "generator_checkpoint.pt")
torch.save({"generator_state": generator.get_state(), "completed": last_completed}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")
print(f"Last completed: {last_completed}")
print(f"Job will resume from Cats/Cartoon onwards.")
