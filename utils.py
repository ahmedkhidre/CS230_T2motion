import numpy as np
import tensorflow as tf
from tqdm import tqdm
import copy
from matplotlib.animation import FuncAnimation 
import datetime
import matplotlib.pyplot as plt


def nn(x_test, y_test, y_pred, x_train, y_train):
    """
    Returns the nearest neighbor metric on time series predictions. The metric
    is a percentage of time when the prediction (by some model) is closer to
    ground truth than a nearest neighbor from the training set.
    """
    assert x_test.shape[0] == y_test.shape[0]
    assert x_test.shape[0] == y_pred.shape[0]
    assert x_train.shape[0] == y_train.shape[0]
    
    model_closer = 0

    for i in tqdm(range(x_test.shape[0])):
        x = x_test[i]
        y_true = y_test[i]
        nn_idx = np.argmin(np.linalg.norm(x_train - x, axis=1))
        nn_distance = np.linalg.norm(y_train[nn_idx] - y_true)
        pred_distance = np.linalg.norm(y_pred[i] - y_true)
        if pred_distance < nn_distance:
            model_closer += 1
    return model_closer / x_test.shape[0]

def diversity(x):
    """
    Computes the diversity of elements of `x` along 0th axis, defined as 1 - average pairwise cosine similarity.
    x - a 2d array
    """
    def cosine_similarity(x, y):
        return np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)
    similarities = []
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            similarities.append(cosine_similarity(x[i], x[j]))
    return 1 - np.mean(similarities)

def create_mlp(input_dim, hidden_dims, output_dim, activation='relu'):
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="input")

    x = inputs
    for i, h in enumerate(hidden_dims):
        x = tf.keras.layers.Dense(h, activation=activation, name=f"dense_{i+1}")(x)
    outputs_flat = tf.keras.layers.Dense(output_dim, activation=None, name="output_flat")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs_flat, name="MLP")
    return model

def load_npz_pairs(npz_path, max_samples=0):
    data = np.load(npz_path, allow_pickle=True)
    z_texts = data["z_texts"]   # shape (N, 384)
    motions = data["motions"]   # dtype=object, each entry (T, J, 3)
    motion_ids = data["motion_ids"] if "motion_ids" in data.files else None

    if max_samples and max_samples > 0:
        z_texts = z_texts[:max_samples]
        motions = motions[:max_samples]
        if motion_ids is not None:
            motion_ids = motion_ids[:max_samples]

    return z_texts, motions, motion_ids


def filter_valid_motions(z_list, motions_list, ids_list=None,
                         num_joints=22, coords=3):
    """
    Filters out invalid motion sequences that do not match (T, num_joints, coords).

    Returns:
        valid_z_list, valid_motions_list, valid_ids_list (or None if no ids)
    """
    valid_z = []
    valid_motions = []
    valid_ids = [] if ids_list is not None else None

    for idx, (z, m) in enumerate(zip(z_list, motions_list)):
        arr = np.array(m)

        # Check dimensionality
        if arr.ndim != 3:
            print(f"[Filter] Skipping sample at index={idx}, shape={arr.shape} (not 3D)")
            continue

        # Check joint and coordinate dimensions
        if arr.shape[1] != num_joints or arr.shape[2] != coords:
            print(f"[Filter] Skipping sample at index={idx}, shape={arr.shape} (bad joint dims)")
            continue

        # Keep sample
        valid_z.append(z)
        valid_motions.append(arr)

        if ids_list is not None:
            valid_ids.append(ids_list[idx])

    if ids_list is not None:
        return np.array(valid_z, dtype=np.float32), valid_motions, valid_ids
    else:
        return np.array(valid_z, dtype=np.float32), valid_motions, None
    
def animate_pose(pose, motion_id):
    pose=copy.deepcopy(pose)
    #SMPL 22-joint skeleton
    edges = [
        (0, 1), (1, 4), (4, 7), (7, 10),
        (0, 2), (2, 5), (5, 8), (8, 11),
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Create lines initially without data
    lines = [ax.plot([], [], [])[0] for _ in range(pose.shape[1] - 1)]
    ax.set(xlim3d=(-1, 3), xlabel='X')
    ax.set(ylim3d=(-1, 1), ylabel='Y')
    ax.set(zlim3d=(-1, 1), zlabel='Z')

    def init(): 
        ax.cla()
        return ax,

    def update_lines(frame_num, pose, lines):
        frame = pose[frame_num]
        for n in range(len(lines)):
            i, j = edges[n]
            x = [frame[i, 0], frame[j, 0]]
            y = [frame[i, 1], frame[j, 1]]
            z = [frame[i, 2], frame[j, 2]]
            lines[n].set_data_3d([z,x,y])
        return lines

    ani = FuncAnimation(fig, update_lines, pose.shape[0], fargs=(pose, lines), interval=100)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ani.save(f'./animations/generate_{motion_id}_{timestamp}.mp4', writer = 'ffmpeg', fps = 30)


def unique_filter(l):
    """
    Generates a filter mask that selects distinct elements of `l`.
    """
    l = np.array(l)
    _, first_occurrence_indices = np.unique(l, return_index=True)
    filter = np.zeros(l.shape[0], dtype=int)
    filter[first_occurrence_indices] = 1
    return filter
