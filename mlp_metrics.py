import tensorflow as tf
import utils 

import autoencoder_utils
import mlp_utils

from utils import unique_filter

TRAIN_NPZ = "./paired_text_motion.npz"
TEST_NPZ  = "./paired_text_motion_val.npz"

MOTION_LEN = 200
NUM_JOINTS = 22
COORDS = 3

POSE_ONLY = True
MODEL_PATH = './models/mlp_pose_only/model.keras' if POSE_ONLY else './models/mlp/model.keras'
USE_VELOCITY_LOSS = True
LAMBDA_VEL = 1


def main():
    model=tf.keras.models.load_model(MODEL_PATH, 
                                     custom_objects={'custom_loss': mlp_utils.custom_loss(MOTION_LEN, USE_VELOCITY_LOSS, LAMBDA_VEL)},
                                     safe_mode=False)

    z_train, motions_train, ids_train = utils.load_npz_pairs(TRAIN_NPZ)
    z_train, motions_train, ids_train = utils.filter_valid_motions(z_train, motions_train, ids_train)
    z_test, motions_test, ids_test = utils.load_npz_pairs(TEST_NPZ)
    z_test, motions_test, ids_test = utils.filter_valid_motions(z_test, motions_test, ids_test)


    X_train, Y_train_flat, _ = mlp_utils.prepare_xy(z_train, motions_train, MOTION_LEN)
    X_test, Y_test_flat, _ = mlp_utils.prepare_xy(z_test, motions_test, MOTION_LEN)

    motions_pred = model.predict(z_test)
    nn_metric = utils.nn(z_test, Y_test_flat, motions_pred, z_train, Y_train_flat)
    print("NN score when comparing 3d motion representations (pose only):\t\t\t\t", nn_metric)

    unique_motions_test = Y_test_flat[unique_filter(ids_test)==1]
    motions_test_diversity = utils.diversity(unique_motions_test)
    print("Diversity of test motions when comparing frames in 3d:\t\t\t\t", motions_test_diversity)

    unique_motions_pred = motions_pred[unique_filter(ids_test)==1, :]
    motions_pred_diversity = utils.diversity(unique_motions_pred)
    print("Diversity of test motion predictions when comparing frames in 3d:\t\t", motions_pred_diversity)

if __name__ == '__main__':
    main()