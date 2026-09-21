# GCLIP-SJR

Complete rice-panicle segmentation pipeline: frozen DINOv2-S/14 features,
synthetic-source diagonal GMM, source-side CLIP semantic assignment, and
Semantic-guided Joint Refinement (SJR). The selected configuration is **560**.
Outputs include native-resolution segmentation and visible panicle-area fraction.

This repository contains the production pipeline and the required source-model
construction steps. Counting, alternative clustering experiments, ablations,
parameter searches, evaluation runners, manuscripts and image datasets are omitted.

## Installation

Python 3.11 is the reference version. Use a virtual environment and install a
PyTorch build appropriate for your device. SJR requires the real pydensecrf C++
extension; install a C++ compiler (MSVC Build Tools with the C++ workload on Windows).

```bash
python -m pip install -r requirements.txt
python -m pip install Cython==0.29.36 setuptools wheel
python -m pip install --no-build-isolation -r requirements-sjr.txt
python download_weights.py --output downloads
```

The downloader checks pinned DINOv2 code and checkpoint hashes. Large pretrained
weights are downloaded from the provider and are not included here. Missing
DenseCRF raises an error; no alternate refinement backend is substituted.

## Run on an image or folder

```bash
python prepare_inputs.py path/to/image_or_folder --output inputs.json
python predict.py --manifest inputs.json --dinov2-repo downloads/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 --checkpoint downloads/dinov2_vits14_pretrain.pth --output predictions --overlay
```

The default is `--refinement sjr`. Use `--refinement none` for raw GCLIP.
Every run requires a new empty output directory. Each input JSON row contains
only `image_id` and `image`; target-mask fields are rejected. Duplicate IDs and
model/semantic hash mismatches also fail explicitly.

Outputs:

- `masks/`: native-size binary panicle masks (0/255).
- `components/`: native-size three-component label maps (0/1/2).
- `overlays/`: optional cyan overlays.
- `area_fractions.csv`: foreground pixels, image pixels and their ratio.
- `inference.json`: model/code hashes, method parameters, environment and records.
- `progress.json`: completed images, retained if a run is interrupted.

Area fraction measures visible projected coverage. It is not physical panicle
area, biomass, yield or instance count. Inference reads no target annotations.

## Pipeline files

| File | Purpose |
|---|---|
| `features.py` | Letterbox, frozen DINO feature extraction and GMM classification |
| `controlled_source_clustering.py` | Source GMM representation, scores and fitting |
| `sjr.py` | Fixed SJR (560), local PAMR and semantic DenseCRF messages |
| `predict.py` | Full target inference and projected-area outputs |
| `prepare_inputs.py`, `download_weights.py` | Input manifests and verified model downloads |
| `extract_source.py`, `fit_source.py`, `calibrate_clip.py` | Offline source-model construction |
| `models/`, `configs/`, `data/` | Compact fitted GMM, hash bindings, prompts and source manifests |
| `third_party/` | Upstream PAMR implementation and Apache-2.0 license |

`data/` contains metadata only. Source masks, target masks, photos, feature caches
and pretrained encoder weights are not bundled. The included GMM is sufficient
for inference; source reconstruction is optional.

## Offline source-model construction

The fixed source resource contains 1,400 synthetic parents and 14,000 correlated
derived views. Obtain its images separately and retain the recorded structure:

```bash
python extract_source.py --source-root path/to/synthetic_rice_200 --dinov2-repo downloads/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 --checkpoint downloads/dinov2_vits14_pretrain.pth --output source_features
python fit_source.py --features source_features/features.npy --output source_gmm.npz --seed 42
python -m pip install -r requirements-calibration.txt
python download_weights.py --output downloads --clip
python calibrate_clip.py --source-root path/to/synthetic_rice_200 --dinov2-repo downloads/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 --checkpoint downloads/dinov2_vits14_pretrain.pth --clip-directory downloads/clip --output semantic_assignment
```

`fit_source.py` implements the K=3 diagonal GMM. K-means++ is used only for
initialization. The final calibration command defaults to the included GMM.
For a newly fitted GMM, create a model JSON using `models/model.json` as a schema,
update its model filename and SHA256, then pass it to `calibrate_clip.py --model`.
Use the generated binding with `predict.py --model`; never assume the new
foreground component is 2. The full feature cache needs about 21.5 GB decimal.
Source image generation itself is not included; prompts alone do not establish
exact regeneration of the fixed synthetic dataset.

Source resource: [Panicle_GMM_data_release](https://pan.baidu.com/s/1_3Zn60ipXP16vOrYWNqwbA), code **1111**.

## Method and reproduction scope

See [the fixed numerical protocol](docs/PROTOCOL.md) and
[verification scope](docs/VERIFICATION.md). Target masks do not train the model,
but labeled development screening was used to select refinement settings.
This is not a claim of annotation-free development or unseen-season validation.

No blanket project license has been specified. Code, data and pretrained models
retain their respective rights; see [third-party notices](THIRD_PARTY_NOTICES.md).
