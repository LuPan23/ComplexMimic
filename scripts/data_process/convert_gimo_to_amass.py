import os
import json
import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# =======================
# 0) HARD-CODED CONFIG
# =======================
DATAROOT = "./data/GIMO"
OUT_ROOT = "./data/GIMO_Motion_Processed/npz_z_up"

FPS   = 30
EVERY = 1


GENDER_STR  = "neutral"
ACTION_TEXT = "GIMO"

# transform json candidates
TRANSFORM_JSON_CANDS = ["transform_info1.json", "transform_infox.json", "transform_info.json"]

# VPoser
VPOSER_SNAPSHOT = "/home/bygpu/Public/2025/12/GIMO/vposer_v1_0/snapshots/TR00_E096.pt"
DEVICE = "cuda"  # "cpu" or "cuda"

# Y-up -> Z-up : rotate +90deg about X
R_YUP_TO_ZUP = np.array(
    [[1, 0, 0],
     [0, 0, -1],
     [0, 1, 0]], dtype=np.float32
)

# =======================
# 1) Generic utils
# =======================
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def np_str_scalar(s: str):
    # keep as numpy scalar string
    return np.array(s)

def _arr(x, n=None, default=None):
    if x is None:
        if default is None:
            return None
        x = default
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if n is not None and x.shape[0] != n:
        raise ValueError(f"Expected {n}, got {x.shape}")
    return x

# =======================
# 2) Find scenes / seqs
# =======================
def list_scenes(dataroot: str):
    scenes = []
    for name in sorted(os.listdir(dataroot)):
        p = os.path.join(dataroot, name)
        if os.path.isdir(p):
            scenes.append(name)
    return scenes

def list_seqs(dataroot: str, scene: str):
    scene_dir = os.path.join(dataroot, scene)
    seqs = []
    for name in sorted(os.listdir(scene_dir)):
        p = os.path.join(scene_dir, name)
        if not os.path.isdir(p):
            continue
        if name == "scene_obj":
            continue
        # valid seq should contain smplx_local
        if os.path.isdir(os.path.join(p, "smplx_local")):
            seqs.append(name)
    return seqs

# =======================
# 3) Load GIMO transforms (per scene+seq)
# =======================
def find_transform_json(dataroot, scene, seq):
    base = os.path.join(dataroot, scene, seq)
    for name in TRANSFORM_JSON_CANDS:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return ""

def load_gimo_transform(dataroot, scene, seq):
    """
    Matches eval_dataset-style:
      T_pose = T_norm @ T_pose2scene
    Both translations divided by scale.
    """
    tf_json = find_transform_json(dataroot, scene, seq)
    if not tf_json:
        raise FileNotFoundError(f"No transform_info*.json under {os.path.join(dataroot, scene, seq)}")

    scene_obj_dir = os.path.join(dataroot, scene, "scene_obj")
    transform_norm_txt = os.path.join(scene_obj_dir, "transform_norm.txt")
    if not os.path.exists(transform_norm_txt):
        raise FileNotFoundError(f"transform_norm.txt not found: {transform_norm_txt}")

    info = json.load(open(tf_json, "r"))
    scale = float(info["scale"])

    T_pose2scene = np.array(info["transformation"], dtype=np.float32)
    T_pose2scene[:3, 3] /= scale

    T_norm = np.loadtxt(transform_norm_txt, dtype=np.float32).reshape(4, 4)
    T_norm[:3, 3] /= scale

    T_pose = (T_norm @ T_pose2scene).astype(np.float32)
    return float(scale), T_norm.astype(np.float32), T_pose

# =======================
# 4) Load frames
# =======================
def list_frame_ids(smplx_local_dir: str):
    pkl_ids = sorted(
        int(os.path.splitext(fn)[0])
        for fn in os.listdir(smplx_local_dir)
        if fn.endswith(".pkl") and os.path.splitext(fn)[0].isdigit()
    )
    if len(pkl_ids) == 0:
        raise FileNotFoundError(f"No *.pkl frames in: {smplx_local_dir}")
    return pkl_ids

def load_gimo_frame_pkl_full(pkl_path: str):
    """
    expected keys:
      orient (3), trans(3), latent(32), optional lhand(45), rhand(45), beta(10)
    """
    data = pickle.load(open(pkl_path, "rb"))
    orient = _arr(data.get("orient", None), 3)
    trans  = _arr(data.get("trans",  None), 3)

    latent = data.get("latent", None)
    latent = _arr(latent, 32) if latent is not None else None

    lhand = _arr(data.get("lhand", None), 45, default=np.zeros((45,), np.float32))
    rhand = _arr(data.get("rhand", None), 45, default=np.zeros((45,), np.float32))
    beta  = data.get("beta", None)
    beta  = _arr(beta, 10) if beta is not None else None

    if orient is None or trans is None:
        raise RuntimeError(f"Missing orient/trans in {pkl_path}")
    if latent is None:
        raise RuntimeError(f"Missing latent in {pkl_path} (VPoser required)")

    return orient, trans, latent, lhand, rhand, beta

# =======================
# 5) VPoser decode (matrot only, batch)
# =======================
@torch.no_grad()
def rotation_matrix_to_angle_axis_torch(rot_mats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    rot_mats: (...,3,3)
    return:   (...,3) axis-angle
    """
    r00 = rot_mats[..., 0, 0]; r01 = rot_mats[..., 0, 1]; r02 = rot_mats[..., 0, 2]
    r10 = rot_mats[..., 1, 0]; r11 = rot_mats[..., 1, 1]; r12 = rot_mats[..., 1, 2]
    r20 = rot_mats[..., 2, 0]; r21 = rot_mats[..., 2, 1]; r22 = rot_mats[..., 2, 2]
    trace = r00 + r11 + r22

    qw = torch.zeros_like(trace)
    qx = torch.zeros_like(trace)
    qy = torch.zeros_like(trace)
    qz = torch.zeros_like(trace)

    cond0 = trace > 0.0
    s0 = torch.sqrt(torch.clamp(trace + 1.0, min=0.0)) * 2.0
    qw0 = 0.25 * s0
    qx0 = (r21 - r12) / torch.clamp(s0, min=eps)
    qy0 = (r02 - r20) / torch.clamp(s0, min=eps)
    qz0 = (r10 - r01) / torch.clamp(s0, min=eps)
    qw = torch.where(cond0, qw0, qw); qx = torch.where(cond0, qx0, qx)
    qy = torch.where(cond0, qy0, qy); qz = torch.where(cond0, qz0, qz)

    cond1 = (~cond0) & (r00 > r11) & (r00 > r22)
    s1 = torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=0.0)) * 2.0
    qw1 = (r21 - r12) / torch.clamp(s1, min=eps)
    qx1 = 0.25 * s1
    qy1 = (r01 + r10) / torch.clamp(s1, min=eps)
    qz1 = (r02 + r20) / torch.clamp(s1, min=eps)
    qw = torch.where(cond1, qw1, qw); qx = torch.where(cond1, qx1, qx)
    qy = torch.where(cond1, qy1, qy); qz = torch.where(cond1, qz1, qz)

    cond2 = (~cond0) & (~cond1) & (r11 > r22)
    s2 = torch.sqrt(torch.clamp(1.0 + r11 - r00 - r22, min=0.0)) * 2.0
    qw2 = (r02 - r20) / torch.clamp(s2, min=eps)
    qx2 = (r01 + r10) / torch.clamp(s2, min=eps)
    qy2 = 0.25 * s2
    qz2 = (r12 + r21) / torch.clamp(s2, min=eps)
    qw = torch.where(cond2, qw2, qw); qx = torch.where(cond2, qx2, qx)
    qy = torch.where(cond2, qy2, qy); qz = torch.where(cond2, qz2, qz)

    cond3 = (~cond0) & (~cond1) & (~cond2)
    s3 = torch.sqrt(torch.clamp(1.0 + r22 - r00 - r11, min=0.0)) * 2.0
    qw3 = (r10 - r01) / torch.clamp(s3, min=eps)
    qx3 = (r02 + r20) / torch.clamp(s3, min=eps)
    qy3 = (r12 + r21) / torch.clamp(s3, min=eps)
    qz3 = 0.25 * s3
    qw = torch.where(cond3, qw3, qw); qx = torch.where(cond3, qx3, qx)
    qy = torch.where(cond3, qy3, qy); qz = torch.where(cond3, qz3, qz)

    qw = torch.clamp(qw, -1.0, 1.0)
    angle = 2.0 * torch.acos(qw)
    sin_half = torch.sqrt(torch.clamp(1.0 - qw * qw, min=0.0))

    k = 1.0 / torch.clamp(sin_half, min=eps)
    axis = torch.stack([qx, qy, qz], dim=-1) * k.unsqueeze(-1)
    aa = axis * angle.unsqueeze(-1)

    small = sin_half < 1e-6
    aa_small = 2.0 * torch.stack([qx, qy, qz], dim=-1)
    aa = torch.where(small.unsqueeze(-1), aa_small, aa)
    return aa

def try_load_vposer(snapshot_pt: str, device: str):
    from human_body_prior.tools.model_loader import load_vposer
    if not os.path.isfile(snapshot_pt) or not snapshot_pt.endswith(".pt"):
        raise FileNotFoundError(f"VPOSER snapshot not found: {snapshot_pt}")
    expr_dir = os.path.dirname(os.path.dirname(snapshot_pt))  # .../vposer_v1_0
    vposer, _ = load_vposer(expr_dir, vp_model="snapshot")
    vposer = vposer.to(device).eval()
    print(f"[INFO] VPoser loaded from: {expr_dir} (device={device})")
    return vposer

def _pick_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ["matrot", "pose_body", "body_pose", "rotmat", "Xout"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v
    if isinstance(out, (list, tuple)):
        for v in out:
            if torch.is_tensor(v):
                return v
    raise RuntimeError(f"Unsupported vposer.decode output type: {type(out)}")

@torch.no_grad()
def vposer_decode_batch_to_aa63(vposer, latents_np: np.ndarray) -> np.ndarray:
    """
    latents_np: (T,32)
    return: aa63_np (T,63)
    """
    dev = next(vposer.parameters()).device
    z = torch.from_numpy(np.asarray(latents_np, dtype=np.float32)).to(dev)  # (T,32)

    out = vposer.decode(z, output_type="matrot")
    t = _pick_tensor(out)

    x = t
    # squeeze potential singleton dims like (T,1,J,9) or (T,1,J,3,3)
    while x.ndim >= 3 and x.shape[1] == 1:
        x = x.squeeze(1)

    # normalize to (T,J,3,3)
    if x.ndim == 3 and x.shape[-1] == 9:          # (T,J,9)
        Tn, J, _ = x.shape
        x = x.view(Tn, J, 3, 3)
    elif x.ndim == 4 and x.shape[-2:] == (3, 3):  # (T,J,3,3)
        pass
    elif x.ndim == 2 and x.shape[1] % 9 == 0:     # (T,J*9)
        Tn = x.shape[0]
        J = x.shape[1] // 9
        x = x.view(Tn, J, 3, 3)
    else:
        raise RuntimeError(f"Unexpected matrot tensor shape: {tuple(x.shape)}")

    if x.shape[1] < 21:
        raise RuntimeError(f"matrot joints J={x.shape[1]} < 21")

    rotm = x[:, :21]  # (T,21,3,3)
    aa = rotation_matrix_to_angle_axis_torch(rotm)  # (T,21,3)
    aa63 = aa.reshape(aa.shape[0], -1).detach().cpu().numpy().astype(np.float32)  # (T,63)
    if aa63.shape[1] != 63:
        raise RuntimeError(f"aa63 wrong shape: {aa63.shape}")
    return aa63

# =======================
# 6) Per-seq export
# =======================
def export_one_seq(scene: str, seq: str, vposer):
    smplx_local_dir = os.path.join(DATAROOT, scene, seq, "smplx_local")

    # load transforms for this seq
    scale, T_norm, T_pose = load_gimo_transform(DATAROOT, scene, seq)

    frame_ids_all = list_frame_ids(smplx_local_dir)
    frame_ids = frame_ids_all[:: max(1, int(EVERY))]
    T = len(frame_ids)
    if T == 0:
        raise RuntimeError("No frames after stride")

    # buffers
    trans_out = np.zeros((T, 3), dtype=np.float32)
    poses_out = np.zeros((T, 156), dtype=np.float32)
    betas_out = np.zeros((10,), dtype=np.float32)
    betas_set = False

    # collect inputs for batch vposer decode
    latents = np.zeros((T, 32), dtype=np.float32)
    lhands  = np.zeros((T, 45), dtype=np.float32)
    rhands  = np.zeros((T, 45), dtype=np.float32)
    orients = np.zeros((T, 3), dtype=np.float32)
    transes = np.zeros((T, 3), dtype=np.float32)

    for i, fid in enumerate(frame_ids):
        pkl_path = os.path.join(smplx_local_dir, f"{fid}.pkl")
        orient, trans, latent, lhand, rhand, beta = load_gimo_frame_pkl_full(pkl_path)

        orients[i] = orient
        transes[i] = trans
        latents[i] = latent
        lhands[i]  = lhand
        rhands[i]  = rhand

        if (not betas_set) and (beta is not None) and beta.shape[0] == 10:
            betas_out[:] = beta.astype(np.float32)
            betas_set = True

    # batch decode body aa63
    aa63_all = vposer_decode_batch_to_aa63(vposer, latents)  # (T,63)

    # per-frame: transform root R/t then write outputs
    for i in range(T):
        # pose space -> canonical scene
        R0 = R.from_rotvec(orients[i]).as_matrix()
        R_s = (T_pose[:3, :3] @ R0).astype(np.float32)

        t0 = transes[i].reshape(3, 1)
        t_s = (T_pose[:3, :3] @ t0 + T_pose[:3, 3:]).reshape(3).astype(np.float32)

        # Y-up -> Z-up
        R_s = (R_YUP_TO_ZUP @ R_s).astype(np.float32)
        t_s = (R_YUP_TO_ZUP @ t_s.reshape(3, 1)).reshape(3).astype(np.float32)

        t_s = t_s + np.array([0.0, 0.2, -0.4], dtype=np.float32)

        trans_out[i] = t_s

        # root rotvec
        poses_out[i, 0:3] = R.from_matrix(R_s).as_rotvec().astype(np.float32)

        # body
        poses_out[i, 3:66] = aa63_all[i].astype(np.float32)

        # hands
        poses_out[i, 66:111]  = lhands[i]
        poses_out[i, 111:156] = rhands[i]

    out_dir = os.path.join(OUT_ROOT, scene, seq)
    ensure_dir(out_dir)

    out_npz = os.path.join(out_dir, f"{scene}_{seq}_fps{FPS}.npz")
    np.savez_compressed(
        out_npz,
        trans=trans_out.astype(np.float32),
        gender=np_str_scalar(GENDER_STR),
        mocap_framerate=np.int64(FPS),
        betas=betas_out.astype(np.float32),
        poses=poses_out.astype(np.float32),
        seg_start_incl=np.int32(0),
        seg_end_incl=np.int32(T - 1),
        seg_length=np.int32(T),
        action_text=np_str_scalar(ACTION_TEXT),
    )
    return out_npz, T, scale

# =======================
# 7) Main (batch)
# =======================
def main():
    ensure_dir(OUT_ROOT)

    if not os.path.isdir(DATAROOT):
        raise NotADirectoryError(f"DATAROOT not found: {DATAROOT}")

    # load vposer once
    vposer = try_load_vposer(VPOSER_SNAPSHOT, device=DEVICE)

    scenes = list_scenes(DATAROOT)
    print(f"[INFO] Scenes: {len(scenes)} under {DATAROOT}")

    ok = 0
    skipped = 0

    for scene in scenes:
        scene_dir = os.path.join(DATAROOT, scene)
        scene_obj_dir = os.path.join(scene_dir, "scene_obj")
        if not os.path.isdir(scene_obj_dir):
            continue  # non-scene folder or irrelevant
        if not os.path.exists(os.path.join(scene_obj_dir, "transform_norm.txt")):
            print(f"[SKIP] {scene}: missing scene_obj/transform_norm.txt")
            skipped += 1
            continue

        seqs = list_seqs(DATAROOT, scene)
        if len(seqs) == 0:
            continue

        for seq in seqs:
            try:
                out_npz, T, scale = export_one_seq(scene, seq, vposer)
                print(f"[OK] {scene}/{seq}: T={T}, scale={scale:.6f} -> {out_npz}")
                ok += 1
            except Exception as e:
                print(f"[SKIP] {scene}/{seq}: {e}")
                skipped += 1

    print(f"[DONE] ok={ok}, skipped={skipped}, out_root={OUT_ROOT}")

if __name__ == "__main__":
    main()
