#!/bin/bash
# Script to resolve git-annex label stubs from backup data
# Run this from the project root: bash scripts/setup_data.sh

BACKUP_LABELS="/Users/L091835/Projects/AI/CNS/.backup/spine-generic/derivatives/labels"
DATA_LABELS="/Users/L091835/Projects/AI/CNS/data/spine-generic/derivatives/labels"

echo "Copying real label files from backup to data directory..."
echo "Source: $BACKUP_LABELS"
echo "Dest:   $DATA_LABELS"

count=0
for src in "$BACKUP_LABELS"/sub-*/anat/*_T2w_label-SC_seg.nii.gz; do
    if [ -f "$src" ] && [ "$(wc -c < "$src")" -gt 1000 ]; then
        # Extract the relative path
        rel="${src#$BACKUP_LABELS/}"
        dest="$DATA_LABELS/$rel"
        dest_dir="$(dirname "$dest")"

        mkdir -p "$dest_dir"
        cp -f "$src" "$dest"
        count=$((count + 1))
    fi
done

echo "Copied $count label files."
echo ""
echo "Verifying..."
verified=$(find "$DATA_LABELS" -name "*T2w_label-SC_seg*" -size +1000c | wc -l)
echo "Label files > 1000 bytes: $verified"
