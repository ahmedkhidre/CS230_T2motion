import numpy as np
import tensorflow as tf
import os

import autoencoder_utils
import utils

FROM_SCRATCH = False

TRAIN_NPZ = "./paired_text_motion.npz"
TEST_NPZ  = "./paired_text_motion_val.npz"

EPOCHS = 100
BATCH_SIZE=512
MAX_MOTION_LENGTH = 200
MIN_MOTION_LENGTH = 50
ENCODING_DIM = 256

POSE_ONLY = True

SAVE_FULL_MODEL_PATH = "./models/autoencoder_pose_only/autoencoder.keras" if POSE_ONLY else  "./models/autoencoder/autoencoder.keras"

def remove_trajectory(X):
    """
    X - (None, None, 22, 3)
    """
    return X - tf.expand_dims(X[:, :, 0, :], axis=2)

def deduplicate(motions):
    prev_motion = motions[0]
    deduplicated_motions = [prev_motion]
    for motion in motions:
        if motion.shape != prev_motion.shape:
            deduplicated_motions.append(motion)
            continue
        if not tf.math.reduce_all(tf.equal(motion,prev_motion)):
            deduplicated_motions.append(motion)
        prev_motion = motion
    return tf.convert_to_tensor(deduplicated_motions)

def prepare_xy(motions):
    padded_motions = tf.convert_to_tensor(
        tf.keras.utils.pad_sequences(motions, padding="post", dtype='float32', maxlen=MAX_MOTION_LENGTH))
    deduplicate(padded_motions)
    if POSE_ONLY:
        padded_motions = remove_trajectory(padded_motions)
    X = tf.reshape(padded_motions[:, 0:MAX_MOTION_LENGTH, :, :], (padded_motions.shape[0], MAX_MOTION_LENGTH, -1))
    return X, X


def main():
    print("Loading data")
    z_train, motions_train, ids_train = utils.load_npz_pairs(TRAIN_NPZ, max_samples=0)
    z_valid, motions_valid, ids_valid = utils.load_npz_pairs(TEST_NPZ, max_samples=0)

    # Filter out motions that are actually just a single frame. These motions have a tensor with 1 fewer dimension
    not_single_frame_fn = np.vectorize(lambda m: len(m.shape) == 3)
    train_selector = not_single_frame_fn(motions_train)
    valid_selector = not_single_frame_fn(motions_valid)
    motions_train = motions_train[train_selector]
    z_train = z_train[train_selector]

    motions_valid = motions_valid[valid_selector]
    z_valid = z_valid[valid_selector]
    # Filter out training motions that are too short.
    min_length_selector = np.vectorize(lambda m : m.shape[0] >= MIN_MOTION_LENGTH)(motions_train)
    motions_train = motions_train[min_length_selector]
    z_train = z_train[min_length_selector]

    X_train, Y_train = prepare_xy(motions_train)
    X_valid, Y_valid = prepare_xy(motions_valid)

    if not FROM_SCRATCH and os.path.exists(SAVE_FULL_MODEL_PATH):
        print("Loading model (weights or full checkpoint)")
        autoencoder=tf.keras.models.load_model(SAVE_FULL_MODEL_PATH)
    else:
        print("Training from scratch.")
        autoencoder, _, _ = autoencoder_utils.autoencoder_model(MAX_MOTION_LENGTH, ENCODING_DIM, 22 * 3)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, weight_decay=0.004, global_clipnorm=0.001)
        autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])
    print(autoencoder.summary())
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVE_FULL_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,     # Save best weights only
        save_weights_only=False,
        verbose=1
    )

    history = autoencoder.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[checkpoint_cb],
    )

if __name__ == '__main__':
    main()