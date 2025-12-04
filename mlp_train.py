import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
import datetime
import utils
import mlp_utils

FROM_SCRATCH=False # True: start fresh, False: resume from checkpoint

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = f"./fit/{TIMESTAMP}/"

TRAIN_NPZ = "./paired_text_motion.npz"
TEST_NPZ  = "./paired_text_motion_val.npz"

POSE_ONLY = True
SAVE_FULL_MODEL_PATH = "./models/mlp_pose_only/model.keras" if POSE_ONLY else  "./models/mlp/model.keras"

MOTION_LEN = 200         # fixed length for MLP outputs (frames)
NUM_JOINTS = 22
COORDS = 3
OUTPUT_DIM = MOTION_LEN * NUM_JOINTS * COORDS


BATCH_SIZE = 4096
EPOCHS = 30
LEARNING_RATE = 1e-4
HIDDEN_DIMS = [1024, 2048, 8192]  # list: number of neurons per hidden Dense layer

# Loss options
USE_VELOCITY_LOSS = True
LAMBDA_VEL = 1
USE_WEIGHT_DECAY = True
WEIGHT_DECAY = 1e-4

# Training hyperparams
USE_LR_SCHEDULER = False
USE_EARLY_STOPPING = False
USE_LR_LOGGER = False  # only if you added the logger

def initialize_callbacks():
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,   # enables gradients & weights monitoring
        write_graph=False
    )

    # Reduce LR on Plateau
    reduceLR_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    # Early stopping
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    # Save best weights only
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVE_FULL_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    class LRTensorBoard(tf.keras.callbacks.Callback):
        def __init__(self, log_dir):
            super().__init__()
            self.file_writer = tf.summary.create_file_writer(log_dir)

        def on_epoch_end(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            with self.file_writer.as_default():
                tf.summary.scalar("learning_rate", lr, step=epoch)

    lr_logger = LRTensorBoard(log_dir=LOG_DIR)


    callbacks_list = [tensorboard_cb, checkpoint_cb]

    if USE_LR_SCHEDULER:
        callbacks_list.append(reduceLR_cb)

    if USE_EARLY_STOPPING:
        callbacks_list.append(earlystop_cb)

    # Optional: add LR logger if you use it
    if USE_LR_LOGGER:
        callbacks_list.append(lr_logger)
    return callbacks_list

def main():
    print("Loading train npz:", TRAIN_NPZ)
    z_train, motions_train, ids_train = utils.load_npz_pairs(TRAIN_NPZ)
    z_train, motions_train, ids_train = utils.filter_valid_motions(z_train, motions_train, ids_train)
    z_test, motions_test, ids_test = utils.load_npz_pairs(TEST_NPZ)
    z_test, motions_test, ids_test = utils.filter_valid_motions(z_test, motions_test, ids_test)

    print("Preparing training tensors...")
    X_train, Y_train_flat, Y_train_3d = mlp_utils.prepare_xy(z_train, motions_train, MOTION_LEN, POSE_ONLY)
    print("Preparing test tensors...")
    X_test, Y_test_flat, Y_test_3d = mlp_utils.prepare_xy(z_test, motions_test, MOTION_LEN, POSE_ONLY)

    if not FROM_SCRATCH and os.path.exists(SAVE_FULL_MODEL_PATH):
        print("Loading model (weights or full checkpoint)")
        # model=tf.keras.models.load_model(SAVE_FULL_MODEL_PATH)
        model=tf.keras.models.load_model(SAVE_FULL_MODEL_PATH,
                                         custom_objects={'custom_loss': mlp_utils.custom_loss(MOTION_LEN, USE_VELOCITY_LOSS, LAMBDA_VEL)},
                                         safe_mode=False)
    else:
        print("training from scratch.")
        model = utils.create_mlp(X_train.shape[1], HIDDEN_DIMS, OUTPUT_DIM)
        optimizer = AdamW(learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY) #<-- gradient clipping
        model.compile(optimizer=optimizer, loss=mlp_utils.custom_loss(MOTION_LEN, USE_VELOCITY_LOSS, LAMBDA_VEL), metrics=["mse"])
        pass  # do nothing, use existing model

    print(model.summary())

    tf.config.run_functions_eagerly(False)  # for performance
    history = model.fit(
        X_train, Y_train_flat,
        validation_data=(X_test, Y_test_flat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=initialize_callbacks(),
        verbose=1
    )

if __name__ == '__main__':
    main()