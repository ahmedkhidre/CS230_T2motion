import numpy as np
import tensorflow as tf
import utils 

import rnn_utils
from utils import unique_filter

TRAIN_NPZ = "./paired_text_motion.npz"
TEST_NPZ  = "./paired_text_motion_val.npz"

MOTION_LEN = 200         # fixed length for MLP outputs (frames)
NUM_JOINTS = 22
COORDS = 3

POSE_ONLY = True
MODEL_PATH = './models/rnn_2_heads/rnn.keras'

USE_VELOCITY_LOSS = True
LAMBDA_VEL = 1

# Indicates that the model was trained with normalization and therefore the outputs need to be denormalized
USE_NORMALIZATION = True


def main():
    model = tf.keras.models.load_model(MODEL_PATH,
        custom_objects={'pose_loss_with_vel': rnn_utils.pose_loss_with_vel(USE_VELOCITY_LOSS, LAMBDA_VEL),
                        'make_time_indices': rnn_utils.make_time_indices(MOTION_LEN)},
        safe_mode=False
    )

    print("Loading data")
    z_train, motions_train, ids_train = utils.load_npz_pairs(TRAIN_NPZ)
    z_train, motions_train, ids_train = utils.filter_valid_motions(z_train, motions_train, ids_train)
    z_test, motions_test, ids_test = utils.load_npz_pairs(TEST_NPZ)
    z_test, motions_test, ids_test = utils.filter_valid_motions(z_test, motions_test, ids_test)


    pose_mean, pose_std, rootvel_mean, rootvel_std = rnn_utils.compute_pose_rootvel_stats(
        motions_train,
        target_len=MOTION_LEN)

    # Load the data unnormalized. Since we will denormalize the model predictions also.
    X_train, Y_pose_train, Y_root_train = rnn_utils.prepare_xy_multi(
        z_train, motions_train,
        pose_mean, pose_std,
        rootvel_mean, rootvel_std,
        motion_len=MOTION_LEN,
        use_normalization=False
    )

    X_test, Y_pose_test, Y_root_test = rnn_utils.prepare_xy_multi(
        z_test, motions_test,
        pose_mean, pose_std,
        rootvel_mean, rootvel_std,
        motion_len=MOTION_LEN,
        use_normalization=False
    )

    print("Calculating model predictions")
    pred_pose_out, pred_rootvel_out = model.predict(X_test)   # (1,T,J,3), (1,T,3)
    if USE_NORMALIZATION:
        # denormalize root-relative pose
        pred_pose_out = (
            pred_pose_out * pose_std[0, 0, 0, :] + pose_mean[0, 0, 0, :]
        )  # (T,J,3)

        # denormalize root velocities
        pred_rootvel_out = (
            pred_rootvel_out * rootvel_std[0, 0, :] + rootvel_mean[0, 0, :]
        )  # (T,3)
    pose_pred = np.reshape(pred_pose_out, (pred_pose_out.shape[0], -1))
    pose_test = np.array(tf.reshape(Y_pose_test, (Y_pose_test.shape[0], -1)))
    pose_train = np.array(tf.reshape(Y_pose_train, (Y_pose_train.shape[0], -1)))

    nn_metric = utils.nn(X_test, pose_test, pose_pred, X_train, pose_train)
    print("NN score when comparing 3d motion representations (pose only):\t\t\t\t", nn_metric)

    unique_motions_test = np.reshape(Y_pose_test, (Y_pose_test.shape[0], -1))[unique_filter(ids_test)==1, :]
    motions_test_diversity = utils.diversity(unique_motions_test)
    print("Diversity of test motions when comparing frames in 3d (pose only):\t\t\t\t", motions_test_diversity)

    unique_motions_pred = pose_pred[unique_filter(ids_test)==1, :]
    motions_pred_diversity = utils.diversity(unique_motions_pred)
    print("Diversity of test motion predictions when comparing frames in 3d (pose only):\t\t", motions_pred_diversity)

if __name__ == '__main__':
    main()