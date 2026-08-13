# Data Directory

This directory contains dataset symlinks/downloads. Data files are NOT committed to version control.

## Setup Instructions

### 1. Spine-Generic (Primary Dataset)

```bash
# Requires git-annex
brew install git-annex  # macOS

# Clone (metadata only, ~small)
git clone https://github.com/spine-generic/data-multi-subject.git data/spine-generic

# Download a development subset
cd data/spine-generic
git annex get sub-amu01 sub-amu05 sub-balgrist01 sub-balgrist02 \
  sub-stanford02 sub-stanford05 sub-mgh01 sub-mgh02 \
  sub-tehranS01 sub-ubc03 sub-ubc04 sub-ucdavis03 sub-unf07

# Full download when ready (~26 GB)
# git annex get .
```

### 2. MSD Hippocampus (Secondary — Architecture Validation)

Download from: http://medicaldecathlon.com/ → Task04_Hippocampus.tar
Extract to: `data/msd_hippocampus/Task04_Hippocampus/`

### 3. Sass 2017 Reference Geometry

Download STL/OBJ from supplementary of DOI: 10.1186/s12987-017-0085-y
Place in: `data/reference_geometry/`
