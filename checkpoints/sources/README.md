# Source fine-tunes

The 48 checkpoints every descriptor of the paper is read from: six vision-language families
(`clip`, `openclip`, `metaclip`, `siglip`, `siglip2`, `evaclip`) fine-tuned on eight tasks
(`eurosat`, `dtd`, `cars`, `gtsrb`, `mnist`, `resisc45`, `svhn`, `sun397`), named
`<model>_<task>.pt`, 1.6 MB each (74 MB in total). `SHA256SUMS` lists their checksums.

Each file is the Hugging Face PEFT state dict of a LoRA of rank 4 and scaling factor 8 on the
query and value projections of every attention block of the vision tower (48 adapted modules
for a ViT-L), as written by `peft.get_peft_model_state_dict`. Load one with

```python
from pta import config as C, models as M
model, proc = M.load("clip")
M.attach_lora(model, C.SOURCE_RANK, C.SOURCE_ALPHA)
M.load_lora(model, C.source_path("clip", "eurosat"))
```

Training (`pta/finetune.py`, cross-entropy over the task's class-name prompts, text tower
frozen, AdamW with a constant learning rate, the task's augmented training split):

| Sources | Learning rate | Batch | Weight decay | Epochs |
|:--|:--|:--|:--|:--|
| EuroSAT and DTD of CLIP, OpenCLIP, MetaCLIP, SigLIP | 3e-4 | 64 | 1e-4 | 3 |
| the other 40 | 1e-4 | 32 | 0.01 | 3 (SUN397: 1) |

Each adapter is also subject to the license of the pre-trained model it adapts (MIT for OpenCLIP
and EVA-CLIP, Apache 2.0 for SigLIP and SigLIP2, the OpenAI CLIP release's license for CLIP,
CC BY-NC 4.0 for MetaCLIP).

The runs were not seeded; `python -m experiments.train_sources` retrains a source with the same
recipe (into `$PTA_CKPT/sources_retrained`), reproducing the paper's numbers up to the
seed-to-seed variation the paper's appendix measures.
