import os
import tensorflow as tf
import tensorflow.keras.models # Necessary to load the model
from sentence_transformers import SentenceTransformer
import numpy as np

import rnn_utils
import utils


# Necessary to avoid the warning:
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = './models/rnn_2_heads/rnn.keras'
NORM_STATS = "./motion_norm_stats.npz"

MOTION_LEN = 200         # fixed length for MLP outputs (frames)
NUM_JOINTS = 22
COORDS = 3

# Indicates that the model was trained with normalization and therefore the outputs need to be denormalized
USE_NORMALIZATION = True
ROOT_TRAJECTORY = False 

USE_VELOCITY_LOSS = True
LAMBDA_VEL = 1

INPUT = "person has arms extended to side of body shoulder height then moves both hands into centre and holds together"


if __name__ == '__main__':
    text_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    embedding = text_transformer.encode(INPUT)
    stats = np.load(NORM_STATS)
    pose_mean     = stats["pose_mean"]
    pose_std      = stats["pose_std"]
    rootvel_mean  = stats["rootvel_mean"]
    rootvel_std   = stats["rootvel_std"]
    print("Loaded normalization stats from", NORM_STATS)

    model = tf.keras.models.load_model(MODEL_PATH,
                                       custom_objects={'pose_loss_with_vel': rnn_utils.pose_loss_with_vel(USE_VELOCITY_LOSS, LAMBDA_VEL),
                                                       'make_time_indices': rnn_utils.make_time_indices(MOTION_LEN)},
                                       safe_mode=False)
    pred_pose_out, pred_rootvel_out = model.predict(embedding[np.newaxis, :])

    if USE_NORMALIZATION:
        # denormalize root-relative pose
        pred_pose_out = (
            pred_pose_out * pose_std[0, 0, 0, :] + pose_mean[0, 0, 0, :]
        )  # (T,J,3)

        # denormalize root velocities
        pred_rootvel_out = (
            pred_rootvel_out * rootvel_std[0, 0, :] + rootvel_mean[0, 0, :]
        )  # (T,3)

    if ROOT_TRAJECTORY:
        # calculate root trajectory from root velocities
        root_pred = np.zeros_like(pred_rootvel_out, dtype=np.float32)  # (T,3)
        for t in range(1, MOTION_LEN):
            root_pred[t] = root_pred[t - 1] + pred_rootvel_out[t]
        root_pred_expanded = root_pred[:, None, :]                   # (T,1,3)
        pred_world = pred_pose_out + root_pred_expanded              # (T,22,3)
        print("Predicted motion (world coords) shape:", pred_world.shape)
    else:
        pred_world = pred_pose_out

    filename = utils.animate_pose(np.reshape(pred_world, (MOTION_LEN, NUM_JOINTS, COORDS)),  "example")
    print("Motion generated: ", filename)
