#!/usr/bin/env python3
"""
Download the HCP-MMP1.0 Glasser-360 parcel index and the Cole-Anticevic
12-network partition dlabel into data/_atlas/. These files are used by s01
for real parcel-level timeseries extraction and by s03 for mapping each
parcel to one of the seven Yeo-like networks (via config.COLE_TO_YEO7).

Source: https://github.com/ColeLab/ColeAnticevicNetPartition
License: see upstream repository.
"""
import sys
from pathlib import Path
import urllib.request

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from configs import config

FILES = {
    "Glasser360Indices_LR.dscalar.nii":
        "https://github.com/ColeLab/ColeAnticevicNetPartition/raw/master/"
        "Glasser360Indices_LR.dscalar.nii",
    "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii":
        "https://github.com/ColeLab/ColeAnticevicNetPartition/raw/master/"
        "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii",
    # Volumetric Glasser-360 in MNI152 (PennLINC/xcpEngine), used only for the
    # glass-brain figures; its parcel order is R then L (see build_vertical_glassbrain).
    "glasser360MNI.nii.gz":
        "https://raw.githubusercontent.com/PennLINC/xcpEngine/master/"
        "atlas/glasser360/glasser360MNI.nii.gz",
}


def main():
    atlas_dir = config.ATLAS_DIR
    atlas_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        out = atlas_dir / name
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {out} already present ({out.stat().st_size} bytes)")
            continue
        print(f"[fetch] {name}")
        urllib.request.urlretrieve(url, str(out))
        print(f"        -> {out} ({out.stat().st_size} bytes)")
    print("done.")


if __name__ == "__main__":
    main()
