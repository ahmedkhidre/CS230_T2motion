import os
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
import datetime
import utils
import rnn_utils

FROM_SCRATCH = False
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = f"./fit/{TIMESTAMP}/"

SAVE_FULL_MODEL_PATH = f"./models/rnn_2_heads/rnn.keras" # saving full model checkpoint if from scratch=True
TRAIN_NPZ = "./paired_text_motion.npz"
TEST_NPZ  = "./paired_text_motion_val.npz"

# Data loading options
MAX_TRAIN_SAMPLES = 0    # 0 = use all; otherwise use first N pairs
MAX_TEST_SAMPLES = 0  # 0 = use all; otherwise use first N pairs

MOTION_LEN = 200         # fixed length for MLP outputs (frames)
NUM_JOINTS = 22
COORDS = 3
OUTPUT_DIM = MOTION_LEN * NUM_JOINTS * COORDS


# Training hyperparams
D_TIME = 32   # Dimension of learnable time embeddings (e.g. 16/32/64)

BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-4
HIDDEN_DIMS = [256,256]  # list: number of neurons per hidden Dense layer

USE_NORMALIZATION = True  #normalize data to zero mean and unity variance

# Loss options
USE_VELOCITY_LOSS = True
LAMBDA_VEL = 1
LAMBDA_ROOT = 1.0  # weight for root-velocity loss vs pose loss (tune if needed)
WEIGHT_DECAY = 1e-4

# Training Monitoring and Callbacks
USE_LR_SCHEDULER = False
USE_EARLY_STOPPING = True
USE_LR_LOGGER = False # learning rate logger callback

def create_model(input_dim, motion_len, time_embedding_dim, hidden_dims):
    # ---------------------------
    # Inputs: sentence embedding z_text (384-d)
    # ---------------------------
    inputs = tf.keras.Input(
        shape=(input_dim,),  # 384 for all-MiniLM
        dtype=tf.float32,
        name="z_text"
    )

    # 1) Repeat text embedding across time -> [B, T, CAP_DIM]
    z_seq = tf.keras.layers.RepeatVector(motion_len, name="repeat_text")(inputs)

    time_indices = tf.keras.layers.Lambda(
        rnn_utils.make_time_indices(motion_len),
        name="time_indices"
    )(inputs)  # [B, T]

    time_embedding_layer = tf.keras.layers.Embedding(
        input_dim=motion_len,   # one embedding per frame index
        output_dim=time_embedding_dim,
        name="time_embedding"
    )

    time_seq = time_embedding_layer(time_indices)  # [B, T, D_TIME]

    # 3) Concatenate text + time -> [B, T, CAP_DIM + D_TIME]
    decoder_input = tf.keras.layers.Concatenate(
        axis=-1,
        name="concat_text_time"
    )([z_seq, time_seq])

    # 4) Stacked GRU decoder over time using HIDDEN_DIMS

    # Learn initial hidden state from caption only
    init_state = tf.keras.layers.Dense(
        hidden_dims[0],           # same size as first GRU layer
        activation="tanh",
        name="init_state_from_caption"
    )(inputs)                     # inputs is z_text: (B, CAP_DIM)

    # First GRU layer uses learned initial_state
    x = tf.keras.layers.GRU(
        hidden_dims[0],
        return_sequences=True,
        name="gru_layer_1"
    )(decoder_input, initial_state=init_state)

    # Additional GRU layers (if any) stack on top, default zero init is fine
    for i, h in enumerate(hidden_dims[1:], start=2):
        x = tf.keras.layers.GRU(
            h,
            return_sequences=True,
            name=f"gru_layer_{i}"
        )(x)


    # 5) Two heads:
    #    - pose_out:    (B, T, 22, 3)   root-relative pose
    #    - rootvel_out: (B, T, 3)       root velocities

    # Pose head
    pose_out_flat = tf.keras.layers.Dense(
        NUM_JOINTS * COORDS,
        name="pose_frame_dense"
    )(x)                   # (B, T, 66)

    pose_out = tf.keras.layers.Reshape(
        (motion_len, NUM_JOINTS, COORDS),
        name="pose_out"
    )(pose_out_flat)       # (B, T, 22, 3)

    # Root-velocity head
    rootvel_out = tf.keras.layers.Dense(
        COORDS,
        name="rootvel_out"
    )(x)                   # (B, T, 3)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[pose_out, rootvel_out],
        name="GRU_TimeCond_Motion_Decoder"
    )
    return model

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
    z_train, motions_train, ids_train = utils.load_npz_pairs(TRAIN_NPZ, max_samples=MAX_TRAIN_SAMPLES)
    print("Filtering training motions...")
    z_train, motions_train, ids_train = utils.filter_valid_motions(z_train, motions_train, ids_train)
    print("Final train samples:", len(z_train))

    print("Loading test npz:", TEST_NPZ)
    z_test, motions_test, ids_test = utils.load_npz_pairs(TEST_NPZ, max_samples=MAX_TEST_SAMPLES)

    print("Filtering test motions...")
    z_test, motions_test, ids_test = utils.filter_valid_motions(z_test, motions_test, ids_test)
    print("Final test samples:", len(z_test))
    
    pose_mean, pose_std, rootvel_mean, rootvel_std = rnn_utils.compute_pose_rootvel_stats(
        motions_train,
        target_len=MOTION_LEN
    )
    print("Preparing training tensors...")
    X_train, Y_pose_train, Y_root_train = rnn_utils.prepare_xy_multi(
        z_train, motions_train,
        pose_mean, pose_std,
        rootvel_mean, rootvel_std,
        motion_len=MOTION_LEN,
        use_normalization=USE_NORMALIZATION
    )

    print("Preparing test tensors...")
    X_test, Y_pose_test, Y_root_test = rnn_utils.prepare_xy_multi(
        z_test, motions_test,
        pose_mean, pose_std,
        rootvel_mean, rootvel_std,
        motion_len=MOTION_LEN,
        use_normalization=USE_NORMALIZATION
    )

    if not FROM_SCRATCH and os.path.exists(SAVE_FULL_MODEL_PATH):
        print(f"Loading model from: {SAVE_FULL_MODEL_PATH}")
        
        model = tf.keras.models.load_model(SAVE_FULL_MODEL_PATH,
            custom_objects={'pose_loss_with_vel': rnn_utils.pose_loss_with_vel(USE_VELOCITY_LOSS, LAMBDA_VEL),
                            'make_time_indices': rnn_utils.make_time_indices(MOTION_LEN)},
            safe_mode=False
        )
    else:
        print("Training from scratch")
        model = create_model(X_train.shape[1], MOTION_LEN, D_TIME, HIDDEN_DIMS)

        model
        optimizer = AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        model.compile(
            optimizer=optimizer,
            loss=[rnn_utils.pose_loss_with_vel(USE_VELOCITY_LOSS, LAMBDA_VEL), "mse"],      # [pose_out, rootvel_out]
            loss_weights=[1.0, LAMBDA_ROOT]
        )
    print(model.summary())

    tf.config.run_functions_eagerly(False)  # for performance
    history = model.fit(
        X_train,
        [Y_pose_train, Y_root_train],
        validation_data=(X_test, [Y_pose_test, Y_root_test]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=initialize_callbacks(),
        verbose=1
    )
if __name__ == '__main__':
    main()