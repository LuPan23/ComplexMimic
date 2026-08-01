import json
import os

import numpy as np
import trimesh


# =======================
# Configuration
# =======================

DATAROOT = "./data/GIMO"
OUT_ROOT = "./data/GIMO_Scene_Processed/obj_z_up"

OBJ_CANDS = [
    "textured_output.obj",
    "scene_obj.obj",
    "scene.obj",
]

TRANSFORM_JSON_CANDS = [
    "transform_info1.json",
    "transform_infox.json",
    "transform_info.json",
]

# Y-up -> Z-up: rotate +90 degrees around X axis
R_YUP_TO_ZUP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


# =======================
# Utilities
# =======================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_obj(
    obj_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write vertices and triangular faces to an OBJ file."""
    vertices = np.asarray(
        vertices,
        dtype=np.float32,
    ).reshape(-1, 3)

    faces = np.asarray(
        faces,
        dtype=np.int64,
    ).reshape(-1, 3)

    with open(obj_path, "w", encoding="utf-8") as fp:
        fp.write("# Processed scene mesh (Z-up)\n")

        for vertex in vertices:
            fp.write(
                f"v {vertex[0]:.6f} "
                f"{vertex[1]:.6f} "
                f"{vertex[2]:.6f}\n"
            )

        # OBJ face indices start from 1.
        for face in faces:
            fp.write(
                f"f {face[0] + 1} "
                f"{face[1] + 1} "
                f"{face[2] + 1}\n"
            )


def load_T_norm(
    transform_norm_txt: str,
    scale: float,
) -> np.ndarray:
    if abs(scale) < 1e-8:
        raise ValueError(f"Invalid scale: {scale}")

    T_norm = np.loadtxt(
        transform_norm_txt,
        dtype=np.float32,
    ).reshape(4, 4)

    T_norm = T_norm.astype(
        np.float32,
        copy=True,
    )

    T_norm[:3, 3] /= float(scale)

    return T_norm


def find_scene_obj_path(scene_obj_dir: str) -> str:
    for name in OBJ_CANDS:
        obj_path = os.path.join(
            scene_obj_dir,
            name,
        )

        if os.path.isfile(obj_path):
            return obj_path

    return ""


def collect_scales_under_scene(scene_dir: str) -> list:
    """Collect scale values from sequence transform JSON files."""
    scales = []

    for entry in sorted(os.listdir(scene_dir)):
        sequence_dir = os.path.join(
            scene_dir,
            entry,
        )

        if not os.path.isdir(sequence_dir):
            continue

        if entry == "scene_obj":
            continue

        for json_name in TRANSFORM_JSON_CANDS:
            json_path = os.path.join(
                sequence_dir,
                json_name,
            )

            if not os.path.isfile(json_path):
                continue

            try:
                with open(
                    json_path,
                    "r",
                    encoding="utf-8",
                ) as fp:
                    transform_info = json.load(fp)

                if "scale" not in transform_info:
                    continue

                scale = float(transform_info["scale"])

                if np.isfinite(scale) and abs(scale) > 1e-8:
                    scales.append(scale)

            except (
                OSError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                print(
                    f"[WARN] Failed to read "
                    f"{json_path}: {error}"
                )

    return scales


def pick_scale(
    scales: list,
    scene_name: str,
) -> float:
    if not scales:
        print(
            f"[WARN] {scene_name}: no valid scale found, "
            f"fallback scale=1.0"
        )
        return 1.0

    scales_array = np.asarray(
        scales,
        dtype=np.float64,
    )

    if np.ptp(scales_array) > 1e-6:
        unique_scales = sorted({
            round(float(scale), 8)
            for scale in scales_array
        })

        print(
            f"[WARN] {scene_name}: inconsistent scales "
            f"{unique_scales}, use median"
        )

    return float(np.median(scales_array))


def load_mesh_vertices_faces(
    obj_path: str,
) -> tuple:
    loaded = trimesh.load(
        obj_path,
        process=False,
    )

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]

        if not geometries:
            raise ValueError(
                f"No mesh geometry found: {obj_path}"
            )

        mesh = trimesh.util.concatenate(geometries)

    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    else:
        raise TypeError(
            f"Unsupported mesh type: {type(loaded)}"
        )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float32,
    )

    faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Invalid vertices shape: {vertices.shape}"
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Invalid faces shape: {faces.shape}"
        )

    return vertices, faces


def process_scene_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    scale: float,
    T_norm: np.ndarray,
) -> tuple:
    if abs(scale) < 1e-8:
        raise ValueError(f"Invalid scale: {scale}")

    vertices = vertices.astype(
        np.float32,
        copy=True,
    )

    # 1. Scale normalization
    vertices /= float(scale)

    # 2. Apply T_norm
    vertices = (
        T_norm[:3, :3] @ vertices.T
        + T_norm[:3, 3:4]
    ).T

    # 3. Y-up -> Z-up
    vertices = (
        R_YUP_TO_ZUP @ vertices.T
    ).T

    return (
        vertices.astype(np.float32),
        faces.astype(np.int64),
    )


# =======================
# Main
# =======================

def main() -> None:
    if not os.path.isdir(DATAROOT):
        raise NotADirectoryError(
            f"DATAROOT not found: {DATAROOT}"
        )

    ensure_dir(OUT_ROOT)

    scene_names = sorted(
        scene_name
        for scene_name in os.listdir(DATAROOT)
        if os.path.isdir(
            os.path.join(
                DATAROOT,
                scene_name,
            )
        )
    )

    print(
        f"[INFO] Found {len(scene_names)} "
        f"scene folders under: {DATAROOT}"
    )

    success_count = 0
    skipped_count = 0

    for scene_name in scene_names:
        scene_dir = os.path.join(
            DATAROOT,
            scene_name,
        )

        scene_obj_dir = os.path.join(
            scene_dir,
            "scene_obj",
        )

        transform_norm_txt = os.path.join(
            scene_obj_dir,
            "transform_norm.txt",
        )

        if not os.path.isdir(scene_obj_dir):
            print(
                f"[SKIP] {scene_name}: missing scene_obj/"
            )
            skipped_count += 1
            continue

        if not os.path.isfile(transform_norm_txt):
            print(
                f"[SKIP] {scene_name}: "
                f"missing transform_norm.txt"
            )
            skipped_count += 1
            continue

        source_obj_path = find_scene_obj_path(
            scene_obj_dir
        )

        if not source_obj_path:
            print(
                f"[SKIP] {scene_name}: no OBJ found, "
                f"candidates={OBJ_CANDS}"
            )
            skipped_count += 1
            continue

        scales = collect_scales_under_scene(
            scene_dir
        )

        scale = pick_scale(
            scales,
            scene_name,
        )

        try:
            T_norm = load_T_norm(
                transform_norm_txt,
                scale,
            )

            raw_vertices, raw_faces = (
                load_mesh_vertices_faces(
                    source_obj_path
                )
            )

            vertices, faces = process_scene_mesh(
                raw_vertices,
                raw_faces,
                scale,
                T_norm,
            )

        except Exception as error:
            print(
                f"[SKIP] {scene_name}: "
                f"processing failed: {error}"
            )
            skipped_count += 1
            continue

        output_scene_dir = os.path.join(
            OUT_ROOT,
            scene_name,
        )
        ensure_dir(output_scene_dir)

        output_obj_path = os.path.join(
            output_scene_dir,
            f"{scene_name}.obj",
        )

        try:
            write_obj(
                output_obj_path,
                vertices,
                faces,
            )

        except OSError as error:
            print(
                f"[SKIP] {scene_name}: "
                f"failed to save OBJ: {error}"
            )
            skipped_count += 1
            continue

        print(
            f"[OK] {scene_name}: "
            f"vertices={len(vertices)}, "
            f"faces={len(faces)}, "
            f"scale={scale:.8f}, "
            f"saved={output_obj_path}"
        )

        success_count += 1

    print(
        f"[DONE] ok={success_count}, "
        f"skipped={skipped_count}, "
        f"out_root={OUT_ROOT}"
    )


if __name__ == "__main__":
    main()