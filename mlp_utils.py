import tensorflow as tf
import numpy as np
import keras


NUM_JOINTS = 22
COORDS = 3

def pad_or_truncate_motion(motion, target_len):
    T, J, C = motion.shape
    if T == target_len:
        return motion.astype(np.float32)
    if T > target_len:
        return motion[:target_len].astype(np.float32)
    # T < target_len: pad with zeros at end
    pad_len = target_len - T
    last_frame = motion[-1][None, :, :]  # shape (1, J, C)
    pad = np.repeat(last_frame, pad_len, axis=0)  # repeat last frame
    return np.concatenate([motion.astype(np.float32), pad], axis=0)

def prepare_xy(z_list, motions_objectlist, motion_len, pose_only=True):
    n = len(z_list)
    x = np.array(z_list, dtype=np.float32)   # (N, 384)
    y = np.zeros((n, motion_len, NUM_JOINTS, COORDS), dtype=np.float32)
    for i, m in enumerate(motions_objectlist):
        y[i] = pad_or_truncate_motion(m, target_len=motion_len)
    y = tf.convert_to_tensor(y, dtype='float32')
    if pose_only:
        # Subtracts the position of the root joint from all vectors
        y = y - tf.expand_dims(y[:, :, 0, :], axis=2)

    # flatten Y for MLP regression target
    y_flat = tf.reshape(y, (n, -1))
    return x, y_flat, y  # also return 3D Y if needed


# ---------------------------
# Loss: MSE + optional velocity loss + optional weight decay
# ---------------------------
def velocity_loss_from_flat(y_true_flat, y_pred_flat, motion_len):
    # y_true_flat, y_pred_flat: (batch, motion_len * J * C)
    # reshape
    batch = tf.shape(y_true_flat)[0]
    y_true = tf.reshape(y_true_flat, (batch, motion_len, NUM_JOINTS, COORDS))
    y_pred = tf.reshape(y_pred_flat, (batch, motion_len, NUM_JOINTS, COORDS))
    # velocities: difference over time axis
    v_true = y_true[:,1:] - y_true[:,:-1]
    v_pred = y_pred[:,1:] - y_pred[:,:-1]
    return tf.reduce_mean(tf.square(v_pred - v_true))


@keras.saving.register_keras_serializable(package="Project")
def custom_loss(motion_len, use_velocity_loss, lambda_vel):
    """
    Returns a callable loss function that takes (y_true, y_pred)
    and captures the other parameters (motion_len, use_velocity_loss, lambda_vel).
    """
    @keras.saving.register_keras_serializable(package="Project")
    def custom_loss_fn(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        if use_velocity_loss:
            vel = velocity_loss_from_flat(y_true, y_pred, motion_len)
            loss_val = mse + lambda_vel * vel
        else:
            loss_val = mse
        return loss_val
    return custom_loss_fn
