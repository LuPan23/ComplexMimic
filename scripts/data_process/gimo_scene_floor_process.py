import os
import glob
import numpy as np
import open3d as o3d
from typing import Optional

# =======================
# HARD-CODED ROOT PATHS
# =======================
SCENE_MESH_ROOT = "./data/GIMO_Scene_Processed/obj_z_up"
OUT_ROOT        = "./data/GIMO_Scene_Processed/gimo_scene_mesh"

# =======================
# HEIGHT BAND (EDIT THESE)
# =======================
Z_MIN_CUT = 0.05
Z_MAX_CUT = 2.00
MODE = "all"  # "all" or "any"

# =======================
# Preview (optional)
# =======================
DO_PREVIEW = True
RENDER_W, RENDER_H = 1600, 900


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = o3d.io.read_triangle_mesh(path)
    if m.is_empty():
        raise RuntimeError(f"Empty mesh: {path}")

    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    try:
        m.remove_non_manifold_edges()
    except Exception:
        pass
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


def remove_tris_by_z_band(
    mesh: o3d.geometry.TriangleMesh,
    zmin_cut: float,
    zmax_cut: float,
    mode: str = "all",
):
    V = np.asarray(mesh.vertices)   # (Nv,3)
    F = np.asarray(mesh.triangles)  # (Nf,3)

    if V.size == 0 or F.size == 0:
        return mesh, 0, int(F.shape[0])

    z = V[:, 2]
    tri_z = z[F]  # (Nf,3)

    outside = (tri_z < zmin_cut) | (tri_z > zmax_cut)

    if mode == "all":
        kill = np.all(outside, axis=1)
    elif mode == "any":
        kill = np.any(outside, axis=1)
    else:
        raise ValueError("MODE must be 'all' or 'any'")

    keep = ~kill
    F_new = F[keep]

    m2 = o3d.geometry.TriangleMesh()
    m2.vertices = o3d.utility.Vector3dVector(V)
    m2.triangles = o3d.utility.Vector3iVector(F_new)
    m2.remove_unreferenced_vertices()
    m2.remove_degenerate_triangles()
    m2.remove_duplicated_triangles()
    m2.compute_vertex_normals()

    return m2, int(kill.sum()), int(F.shape[0])


def save_preview_png(mesh: o3d.geometry.TriangleMesh, out_png: str):
    try:
        import open3d.visualization.rendering as rendering
    except Exception as e:
        print(f"  [WARN] Offscreen rendering not available, skip preview. ({e})")
        return

    try:
        renderer = rendering.OffscreenRenderer(RENDER_W, RENDER_H)
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 1.0])

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        mesh.compute_vertex_normals()
        scene.add_geometry("mesh", mesh, mat)

        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        radius = float(np.linalg.norm(extent)) * 0.6 + 1e-6

        eye = center + np.array([radius, -radius, radius])
        up = np.array([0.0, 0.0, 1.0])

        renderer.setup_camera(60.0, bbox, center)
        scene.camera.look_at(center, eye, up)

        img = renderer.render_to_image()
        o3d.io.write_image(out_png, img)
        renderer.release()
        print(f"  [OK] Preview saved: {out_png}")
    except Exception as e:
        print(f"  [WARN] Offscreen render failed, skip preview. ({e})")


def pick_input_obj(scene_dir: str, scene_name: str) -> Optional[str]:
    cands = [
        os.path.join(scene_dir, f"scene_mesh_{scene_name}.obj"),
        os.path.join(scene_dir, f"{scene_name}.obj"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p

    all_objs = sorted(glob.glob(os.path.join(scene_dir, "*.obj")))
    if len(all_objs) > 0:
        return all_objs[0]
    return None


def main():
    if not os.path.isdir(SCENE_MESH_ROOT):
        raise NotADirectoryError(f"SCENE_MESH_ROOT not found: {SCENE_MESH_ROOT}")

    ensure_dir(OUT_ROOT)

    scene_dirs = sorted([d for d in glob.glob(os.path.join(SCENE_MESH_ROOT, "*")) if os.path.isdir(d)])
    if len(scene_dirs) == 0:
        raise FileNotFoundError(f"No scene folders under: {SCENE_MESH_ROOT}")

    print(f"[INFO] Scenes found: {len(scene_dirs)}")
    print(f"[INFO] Z band keep: [{Z_MIN_CUT:.4f}, {Z_MAX_CUT:.4f}], MODE={MODE}")
    print(f"[INFO] OUT_ROOT: {OUT_ROOT}")

    ok = 0
    skipped = 0
    failed = 0

    for scene_dir in scene_dirs:
        scene = os.path.basename(scene_dir)
        in_obj = pick_input_obj(scene_dir, scene)
        if in_obj is None:
            print(f"[SKIP] {scene}: no .obj in {scene_dir}")
            skipped += 1
            continue

        out_scene_dir = os.path.join(OUT_ROOT, scene)
        ensure_dir(out_scene_dir)

        out_obj = os.path.join(out_scene_dir, f"{scene}.obj")

        print(f"\n[SCENE] {scene}")
        print(f"  in : {in_obj}")

        try:
            mesh = load_mesh(in_obj)

            V = np.asarray(mesh.vertices)
            zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
            print(f"  Z range: [{zmin:.4f}, {zmax:.4f}]")

            mesh2, killed, total = remove_tris_by_z_band(mesh, Z_MIN_CUT, Z_MAX_CUT, MODE)
            print(f"  tris: total={total}, removed={killed}, kept={total - killed}")

            o3d.io.write_triangle_mesh(out_obj, mesh2, write_triangle_uvs=False, write_vertex_normals=False)
            print(f"  out: {out_obj}")


            ok += 1
        except Exception as e:
            print(f"  [FAIL] {scene}: {e}")
            failed += 1

    print("\n[SUMMARY]")
    print(f"  ok      : {ok}")
    print(f"  skipped : {skipped}")
    print(f"  failed  : {failed}")


if __name__ == "__main__":
    main()
