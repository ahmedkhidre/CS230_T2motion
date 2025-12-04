import tensorflow as tf
import numpy as np
import keras

NUM_JOINTS = 22
COORDS = 3

def pose_loss_with_vel_fn(y_true, y_pred, use_velocity_loss, lambda_vel):
    """
    y_*: (B, T, J, 3)
    MSE on pose + optional velocity loss on pose.
    """
    pos_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    if use_velocity_loss:
        vel_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        vel_pred = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        vel_loss = tf.reduce_mean(tf.square(vel_true - vel_pred))
        return pos_loss + lambda_vel * vel_loss
    else:
        return pos_loss
    
def pose_loss_with_vel(use_velocity_loss, lambda_vel):
    return lambda x, y : pose_loss_with_vel_fn(x, y, use_velocity_loss, lambda_vel)

def make_time_indices_fn(z, motion_len):
    batch_size = tf.shape(z)[0]
    time_range = tf.range(motion_len, dtype=tf.int32)  # [T]
    time_range = tf.expand_dims(time_range, axis=0)    # [1, T]
    time_indices = tf.tile(time_range, [batch_size, 1])  # [B, T]
    return time_indices

@keras.saving.register_keras_serializable(package="Project")
def make_time_indices(motion_len):
    @keras.saving.register_keras_serializable(package="Project")
    def make_time_indices_fn(z):
        batch_size = tf.shape(z)[0]
        time_range = tf.range(motion_len, dtype=tf.int32)  # [T]
        time_range = tf.expand_dims(time_range, axis=0)    # [1, T]
        time_indices = tf.tile(time_range, [batch_size, 1])  # [B, T]
        return time_indices
    
    return make_time_indices_fn


def pad_or_truncate_motion(motion, target_len):
    """
    motion: (T, J, 3) -> returns (target_len, J, 3)
    """
    T, J, C = motion.shape
    motion = motion.astype(np.float32)
    if T == target_len:
        return motion
    if T > target_len:
        return motion[:target_len]
    # T < target_len: pad by repeating last frame
    pad_len = target_len - T
    last_frame = motion[-1][None, :, :]  # (1, J, 3)
    pad = np.repeat(last_frame, pad_len, axis=0)
    return np.concatenate([motion, pad], axis=0)


def prepare_xy_multi(z_list,
                     motions_objectlist,
                     pose_mean, pose_std,
                     rootvel_mean, rootvel_std,
                     motion_len,use_normalization):
    """
    Returns:
      X:       (N, 384)
      Y_pose:  (N, T, J, 3)   normalized root-relative pose
      Y_root:  (N, T, 3)      normalized root velocities
    """
    N = len(z_list)
    X = np.array(z_list, dtype=np.float32)  # (N, 384)

    Y_pose = np.zeros((N, motion_len, NUM_JOINTS, COORDS), dtype=np.float32)
    Y_root = np.zeros((N, motion_len, COORDS), dtype=np.float32)

    for i, m in enumerate(motions_objectlist):
        pose_rel, root_vel = extract_pose_and_rootvel(m, target_len=motion_len)

        pose_norm    = (pose_rel - pose_mean) / pose_std
        rootvel_norm = (root_vel - rootvel_mean) / rootvel_std
        if use_normalization:
            Y_pose[i] = pose_norm
            Y_root[i] = rootvel_norm
        else:
            Y_pose[i] = pose_rel
            Y_root[i] = root_vel

    return X, Y_pose, Y_root


def extract_pose_and_rootvel(motion, target_len):
    """
    motion: (T, NUM_JOINTS, 3) in absolute/world coordinates.
    Returns:
      pose_rel: (T, NUM_JOINTS, 3)  -> root-relative pose
      root_vel: (T, 3)              -> root velocity per frame
    """
    motion_fixed = pad_or_truncate_motion(motion, target_len)  # (T,J,3)

    # global root per frame = joint 0
    root = motion_fixed[:, 0, :]                               # (T,3)
    pose_rel = motion_fixed - motion_fixed[:, 0:1, :]          # (T,J,3)

    # root velocities (Δroot); v[0] = 0
    root_vel = np.zeros_like(root)
    root_vel[1:] = root[1:] - root[:-1]
    return pose_rel, root_vel


def compute_pose_rootvel_stats(motions, target_len):
    """
    Compute mean/std for:
      - root-relative pose (per coord)
      - root velocities (per coord)

    motions: list of (T, J, 3) arrays (TRAIN set)
    Returns:
      pose_mean:     (1,1,1,3)
      pose_std:      (1,1,1,3)
      rootvel_mean:  (1,1,3)
      rootvel_std:   (1,1,3)
    """
    pose_coords = []
    rootvel_coords = []

    for m in motions:
        pose_rel, root_vel = extract_pose_and_rootvel(m, target_len)
        pose_coords.append(pose_rel.reshape(-1, COORDS))    # (T*J,3)
        rootvel_coords.append(root_vel.reshape(-1, COORDS)) # (T,3)

    pose_coords = np.concatenate(pose_coords, axis=0)
    rootvel_coords = np.concatenate(rootvel_coords, axis=0)

    pose_mean = pose_coords.mean(axis=0, keepdims=True)         # (1,3)
    pose_std  = pose_coords.std(axis=0, keepdims=True) + 1e-8
    rootvel_mean = rootvel_coords.mean(axis=0, keepdims=True)   # (1,3)
    rootvel_std  = rootvel_coords.std(axis=0, keepdims=True) + 1e-8

    # reshape for broadcasting
    pose_mean    = pose_mean.reshape(1, 1, 1, 3).astype(np.float32)
    pose_std     = pose_std.reshape(1, 1, 1, 3).astype(np.float32)
    rootvel_mean = rootvel_mean.reshape(1, 1, 3).astype(np.float32)
    rootvel_std  = rootvel_std.reshape(1, 1, 3).astype(np.float32)

    return pose_mean, pose_std, rootvel_mean, rootvel_std