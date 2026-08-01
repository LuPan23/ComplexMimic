import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

JSON_PATH = Path("./data_process/gimo/gimo_inference.json")

NPZ_ROOT = Path("./data/GIMO_Motion_Processed/npz_z_up")

OUT_ROOT = Path("./data/GIMO_Motion_Processed/gimo_motion_inference")


def main():
    # Load the motion-segment mapping file.
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    # Create the output directory.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    npz_index = {
        npz_path.stem: npz_path
        for npz_path in NPZ_ROOT.rglob("*.npz")
    }

    success_count = 0
    failed_count = 0

    for segment_name, segment_info in tqdm(
        mapping.items(),
        desc="Splitting motions",
    ):
        parent_motion = segment_info["parent_motion"]

        # The segment interval follows Python slicing:
        # [start, end)
        start = int(segment_info["start"])
        end = int(segment_info["end"])

        source_path = npz_index.get(parent_motion)

        if source_path is None:
            print(f"\n[MISSING] {parent_motion}.npz")
            failed_count += 1
            continue

        try:
            with np.load(source_path, allow_pickle=True) as source_data:
                # Use poses to determine the total number of frames.
                total_frames = source_data["poses"].shape[0]

                # Validate the requested segment range.
                if start < 0 or end > total_frames or start >= end:
                    raise ValueError(
                        f"Invalid range [{start}, {end}), "
                        f"total_frames={total_frames}"
                    )

                output_data = {}

                for key in source_data.files:
                    value = source_data[key]

                    if (
                        value.ndim > 0
                        and value.shape[0] == total_frames
                    ):
                        output_data[key] = value[start:end]
                    else:
                        output_data[key] = value

                segment_length = end - start

                # Reset segment metadata for the new NPZ file.
                output_data["seg_start_incl"] = np.int32(0)
                output_data["seg_end_incl"] = np.int32(
                    segment_length - 1
                )
                output_data["seg_length"] = np.int32(
                    segment_length
                )

            # Save all split motions directly under one folder.
            output_path = OUT_ROOT / f"{segment_name}.npz"

            np.savez_compressed(
                output_path,
                **output_data,
            )

            success_count += 1

        except Exception as error:
            print(f"\n[FAILED] {segment_name}: {error}")
            failed_count += 1

    print(
        f"\n[DONE] success={success_count}, "
        f"failed={failed_count}"
    )
    print(f"[DONE] output={OUT_ROOT}")


if __name__ == "__main__":
    main()