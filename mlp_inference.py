import os
import tensorflow as tf
import tensorflow.keras.models # Necessary to load the model
from sentence_transformers import SentenceTransformer
import numpy as np

import mlp_utils
import utils


# Necessary to avoid the warning:
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

POSE_ONLY = True
MODEL_PATH = './models/mlp_pose_only/model.keras' if POSE_ONLY else './models/mlp/model.keras'

MOTION_LEN = 200         # fixed length for MLP outputs (frames)
NUM_JOINTS = 22
COORDS = 3

USE_VELOCITY_LOSS = True
LAMBDA_VEL = 1
INPUT = "person has arms extended to side of body shoulder height then moves both hands into centre and holds together"


if __name__ == '__main__':
    text_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    embedding = text_transformer.encode(INPUT)


    model=tf.keras.models.load_model(MODEL_PATH, 
                                        custom_objects={'custom_loss': mlp_utils.custom_loss(MOTION_LEN, USE_VELOCITY_LOSS, LAMBDA_VEL)},
                                        safe_mode=False)

    prediction = model.predict(embedding[np.newaxis, :])
    filename = utils.animate_pose(np.reshape(prediction, (MOTION_LEN, NUM_JOINTS, COORDS)),  "example")
    print("Motion generated: ", filename)
