"""
Standalone Training and Next-Activity Prediction Script Using Pre-Split XES Files

This script loads a full XES log (from raw_datasets) and pre-split training, validation, and test
XES files (from split_datasets). It uses PM4Py to load the logs and converts each trace into a list of tokens.
You can choose whether to use:
  - The computed activity embeddings (via one of our intrinsic methods), OR
  - The original one-hot encodings.
Moreover, you can choose which log to compute the embeddings from:
  - "train" for training set,
  - "validation" for validation set,
  - "train_val" for the combination of training and validation.
The chosen representation is then used in the LSTM model.
Only next-activity prediction is performed here.

Author: Modified for standalone use
"""
import os, sys

# -----------------------
# Set cuDNN environment variables (must be set before TensorFlow is imported)
if os.environ.get("MY_CUDNN_SET") != "true":
    os.environ["LD_LIBRARY_PATH"] = "/vol/fob-vol4/mi17/kirchmah/cudnn-8.9.6/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_PRELOAD"] = "/vol/fob-vol4/mi17/kirchmah/cudnn-8.9.6/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn.so.8.9.6"
    os.environ["MY_CUDNN_SET"] = "true"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Now it's safe to import TensorFlow.
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

import copy, random
from datetime import datetime, timedelta
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set GPU memory growth if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

tf.compat.v1.set_random_seed(42)
random.seed(42)
np.random.seed(42)

# -----------------------
# Select Encoding Mode:
# Set encoding_mode to "embedding" or "one_hot"
encoding_mode = "one_hot"  # Change to "one_hot" for one-hot encodings.
use_one_hot = (encoding_mode == "one_hot")
print("Encoding mode:", encoding_mode)
selected_embedding_method = encoding_mode
# -----------------------
# Select Embedding Source:
# Set embedding_source to "train", "validation", or "train_val"
embedding_source = "train_val"  # Options: "train_val", "validation", "train_val"
print("Embedding source:", embedding_source)

# --------------------------------------------------
# HARD-CODED PATHS FOR PRE-SPLIT XES FILES
# --------------------------------------------------
from definitions import ROOT_DIR
path_to_na = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "next_activity_prediction")
full_dataset = os.path.join(path_to_na, "raw_datasets", "SEPSIS.xes.gz")
train_dataset = os.path.join(path_to_na, "split_datasets", "train_SEPSIS.xes.gz")
val_dataset   = os.path.join(path_to_na, "split_datasets", "val_SEPSIS.xes.gz")
test_dataset  = os.path.join(path_to_na, "split_datasets", "test_SEPSIS.xes.gz")

# --------------------------------------------------
# DATA LOADING: Using PM4Py to load XES files and converting each trace to a list of tokens.
# --------------------------------------------------
from pm4py.objects.log.importer.xes import importer as xes_importer

def load_file_xes(file_path):
    log = xes_importer.apply(file_path)
    lines = []       # List of traces (list of tokens)
    timeseqs = []    # List of time differences (since previous event)
    timeseqs2 = []   # Time since start (seconds)
    timeseqs3 = []   # Seconds since midnight for each event
    timeseqs4 = []   # Weekday for each event
    caseids = []
    for trace in log:
        caseid = trace.attributes.get("concept:name", "case_" + str(len(caseids)))
        caseids.append(caseid)
        tokens = []
        times = []
        times2 = []
        times3 = []
        times4 = []
        casestart = None
        lastevent = None
        for event in trace:
            activity = event["concept:name"]
            timestamp = event["time:timestamp"]
            tokens.append(activity)
            if casestart is None:
                casestart = timestamp
                lastevent = timestamp
            diff = (timestamp - lastevent).total_seconds()
            diff2 = (timestamp - casestart).total_seconds()
            midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            diff3 = (timestamp - midnight).total_seconds()
            diff4 = timestamp.weekday()
            times.append(diff)
            times2.append(diff2)
            times3.append(diff3)
            times4.append(diff4)
            lastevent = timestamp
        tokens.append('!')
        lines.append(tokens)
        timeseqs.append(times)
        timeseqs2.append(times2)
        timeseqs3.append(times3)
        timeseqs4.append(times4)
    return lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, caseids

print("Loading full log from:", full_dataset)
lines_full, ts_full, ts2_full, ts3_full, ts4_full, caseids_full = load_file_xes(full_dataset)
print("Loading training log from:", train_dataset)
lines_train, ts_train, ts2_train, ts3_train, ts4_train, caseids_train = load_file_xes(train_dataset)
print("Loading validation log from:", val_dataset)
lines_val, ts_val, ts2_val, ts3_val, ts4_val, caseids_val = load_file_xes(val_dataset)
print("Loading test log from:", test_dataset)
lines_test, ts_test, ts2_test, ts3_test, ts4_test, caseids_test = load_file_xes(test_dataset)

maxlen = max(len(trace) for trace in lines_full)
print("Maximum trace length:", maxlen)
all_times = [t for trace in ts_train for t in trace]
divisor = np.mean(all_times) if all_times else 1
all_times2 = [t for trace in ts2_train for t in trace]
divisor2 = np.mean(all_times2) if all_times2 else 1
print("Time divisor (avg time diff):", divisor)
print("Time divisor2 (avg time since start):", divisor2)

# Build vocabulary for target activities from the full log.
vocab = sorted(set(token for trace in lines_full for token in trace))
target_tokens = copy.copy(vocab)
if '!' in vocab:
    vocab.remove('!')
target_token_indices = {token: i for i, token in enumerate(target_tokens)}
indices_token = {i: token for i, token in enumerate(target_tokens)}

# --------------------------------------------------
# EMBEDDING EXTRACTION FUNCTIONS
# --------------------------------------------------
# If using embedding mode, choose which log to use:
if not use_one_hot:
    # Decide the source of embeddings:
    if embedding_source == "train":
        embedding_input = lines_train
    elif embedding_source == "validation":
        embedding_input = lines_val
    elif embedding_source == "train_val":
        embedding_input = lines_train + lines_val
    else:
        raise ValueError("Unknown embedding_source: choose 'train', 'validation', or 'train_val'.")

    from distances.activity_distances.activity_context_frequency.activity_contex_frequency import get_activity_context_frequency_matrix
    from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import get_activity_activity_co_occurence_matrix

    def get_activity_embeddings_from_val_context_frequency(lines, ngram_size=3):
        log_input = [trace[:-1] for trace in lines]
        alphabet_input = sorted(set(token for trace in log_input for token in trace))
        _, embeddings, _, _, _ = get_activity_context_frequency_matrix(log_input, alphabet_input, ngram_size, bag_of_words=2)
        return embeddings

    def get_activity_embeddings_from_val_activity_cooccurrence(lines, ngram_size=3, bag_of_words=False):
        log_input = [trace[:-1] for trace in lines]
        alphabet_input = sorted(set(token for trace in log_input for token in trace))
        _, embeddings, _, _ = get_activity_activity_co_occurence_matrix(log_input, alphabet_input, ngram_size, bag_of_words)
        return embeddings

    # Select which embedding method to use:
    #selected_embedding_method = "activity_cooccurrence"
    if selected_embedding_method == "context_frequency":
        activity_embeddings = get_activity_embeddings_from_val_context_frequency(embedding_input)
    elif selected_embedding_method == "activity_cooccurrence":
        activity_embeddings = get_activity_embeddings_from_val_activity_cooccurrence(embedding_input)
    else:
        raise ValueError("Unknown embedding method")
    encoding_dim = len(next(iter(activity_embeddings.values())))
    print("Using embedding method:", selected_embedding_method, "with embedding dimension:", encoding_dim)
else:
    # One-hot mode: use vocabulary dimension.
    encoding_dim = len(vocab)
    print("Using original one-hot encodings with dimension:", encoding_dim)

# Total input features = encoding_dim + 5 (for time features)
num_features = encoding_dim + 5

if use_one_hot:
    one_hot_indices = {token: i for i, token in enumerate(vocab)}

# --------------------------------------------------
# VECTORIZATION FUNCTION
# --------------------------------------------------
def vectorize_fold(lines_fold, ts_fold, ts2_fold, ts3_fold, ts4_fold, divisor, divisor2, use_one_hot, encoding_dim, activity_embeddings=None):
    sentences = []
    next_tokens = []
    sent_ts = []
    sent_ts2 = []
    sent_ts3 = []
    sent_ts4 = []
    for tokens, t_seq, t2_seq, t3_seq, t4_seq in zip(lines_fold, ts_fold, ts2_fold, ts3_fold, ts4_fold):
        for i in range(1, len(tokens)):
            sentences.append(tokens[:i])
            sent_ts.append(t_seq[:i])
            sent_ts2.append(t2_seq[:i])
            sent_ts3.append(t3_seq[:i])
            sent_ts4.append(t4_seq[:i])
            next_tokens.append(tokens[i])
    print("Number of sequences:", len(sentences))
    X = np.zeros((len(sentences), maxlen, encoding_dim + 5), dtype=np.float32)
    y_act = np.zeros((len(sentences), len(target_tokens)), dtype=np.float32)
    y_time = np.zeros((len(sentences)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen - len(sentence)
        for t, token in enumerate(sentence):
            if use_one_hot:
                if token in one_hot_indices:
                    X[i, t + leftpad, one_hot_indices[token]] = 1
            else:
                if token in activity_embeddings:
                    X[i, t + leftpad, :encoding_dim] = activity_embeddings[token]
                else:
                    X[i, t + leftpad, :encoding_dim] = np.zeros(encoding_dim)
            X[i, t + leftpad, encoding_dim]     = t + 1
            X[i, t + leftpad, encoding_dim + 1] = sent_ts[i][t] / divisor
            X[i, t + leftpad, encoding_dim + 2] = sent_ts2[i][t] / divisor2
            X[i, t + leftpad, encoding_dim + 3] = sent_ts3[i][t] / 86400
            X[i, t + leftpad, encoding_dim + 4] = sent_ts4[i][t] / 7
        target = next_tokens[i]
        for token in target_tokens:
            if token == target:
                y_act[i, target_token_indices[token]] = 1
        y_time[i] = sent_ts[i][-1] / divisor
    return X, target_token_indices, y_act, y_time

if use_one_hot:
    X_full, _, y_act_full, y_time_full = vectorize_fold(lines_full, ts_full, ts2_full, ts3_full, ts4_full, divisor, divisor2, use_one_hot, encoding_dim)
    X_train, _, y_act_train, y_time_train = vectorize_fold(lines_train, ts_train, ts2_train, ts3_train, ts4_train, divisor, divisor2, use_one_hot, encoding_dim)
    X_val, _, y_act_val, y_time_val = vectorize_fold(lines_val, ts_val, ts2_val, ts3_val, ts4_val, divisor, divisor2, use_one_hot, encoding_dim)
    X_test, _, y_act_test, y_time_test = vectorize_fold(lines_test, ts_test, ts2_test, ts3_test, ts4_test, divisor, divisor2, use_one_hot, encoding_dim)
else:
    X_full, _, y_act_full, y_time_full = vectorize_fold(lines_full, ts_full, ts2_full, ts3_full, ts4_full, divisor, divisor2, use_one_hot, encoding_dim, activity_embeddings)
    X_train, _, y_act_train, y_time_train = vectorize_fold(lines_train, ts_train, ts2_train, ts3_train, ts4_train, divisor, divisor2, use_one_hot, encoding_dim, activity_embeddings)
    X_val, _, y_act_val, y_time_val = vectorize_fold(lines_val, ts_val, ts2_val, ts3_val, ts4_val, divisor, divisor2, use_one_hot, encoding_dim, activity_embeddings)
    X_test, _, y_act_test, y_time_test = vectorize_fold(lines_test, ts_test, ts2_test, ts3_test, ts4_test, divisor, divisor2, use_one_hot, encoding_dim, activity_embeddings)

# --------------------------------------------------
# BUILD THE MODEL
# --------------------------------------------------
print("Building model...")
main_input = Input(shape=(maxlen, num_features), name='main_input')
l1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2, unroll=True)(main_input)
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2, unroll=True)(b1)
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2, unroll=True)(b1)
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(len(target_tokens), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
model = Model(inputs=[main_input], outputs=[act_output, time_output])
opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
              optimizer=opt,
              metrics={"act_output": "acc", "time_output": "mae"})
model.summary()

# --------------------------------------------------
# DIRECTORIES FOR MODELS AND RESULTS
# --------------------------------------------------
import distutils.dir_util
distutils.dir_util.mkpath("models")
distutils.dir_util.mkpath("results")
best_model = "models/" + os.path.basename(full_dataset).replace(".xes.gz", "") + "_" + selected_embedding_method + ".h5"
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=0.0001)

# --------------------------------------------------
# TRAINING & NEXT-ACTIVITY PREDICTION EVALUATION
# --------------------------------------------------
do_train = True
do_test = True

if do_train:
    model.fit(X_train, {'act_output': y_act_train, 'time_output': y_time_train},
              validation_data=(X_val, {"act_output": y_act_val, "time_output": y_time_val}),
              verbose=1, callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=maxlen, epochs=200)

if do_test:
    model.load_weights(best_model)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
                  optimizer=opt,
                  metrics={"act_output": "acc", "time_output": "mae"})
    metrics = model.evaluate(X_test, {'act_output': y_act_test, 'time_output': y_time_test},
                             verbose=1, batch_size=maxlen)
    preds = model.predict([X_test])
    y_act_pred_probs = preds[0]
    y_act_pred = np.argmax(y_act_pred_probs, axis=1)
    y_act_true = np.argmax(y_act_test, axis=1)
    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
    def calculate_brier_score(y_pred, y_true):
        return np.mean(np.sum((y_true - y_pred)**2, axis=1))
    result_file = "results/" + os.path.basename(full_dataset).replace(".xes.gz", "") + "_" + selected_embedding_method + "_next_event.log"
    with open(result_file, "w") as file:
        for metric, name in zip(metrics, model.metrics_names):
            if name == "time_output_mae":
                file.write("mae_in_days: " + str(metric * (divisor / 86400)) + "\n")
            else:
                file.write(str(name) + ": " + str(metric) + "\n")
        acc = accuracy_score(y_act_true, y_act_pred)
        mcc = matthews_corrcoef(y_act_true, y_act_pred)
        precision = precision_score(y_act_true, y_act_pred, average="weighted")
        recall = recall_score(y_act_true, y_act_pred, average="weighted")
        f1 = f1_score(y_act_true, y_act_pred, average="weighted")
        brier_score = calculate_brier_score(y_act_pred_probs, y_act_test)
        file.write("\nACC Sklearn: " + str(acc))
        file.write("\nMCC: " + str(mcc))
        file.write("\nBrier score: " + str(brier_score))
        file.write("\nWeighted Precision: " + str(precision))
        file.write("\nWeighted Recall: " + str(recall))
        file.write("\nWeighted F1: " + str(f1))
    with open("results/raw_" + os.path.basename(full_dataset).replace(".xes.gz", "") + "_" + selected_embedding_method + ".txt", "w") as raw_file:
        raw_file.write("prefix_length;ground_truth;predicted;prediction_probs\n")
        for X_inst, true_label, pred, pred_probs in zip(X_test, y_act_true, y_act_pred, y_act_pred_probs):
            raw_file.write(str(np.count_nonzero(np.sum(X_inst, axis=-1))) + ";" + str(true_label) + ";" + str(pred) + ";" +
                           np.array2string(pred_probs, separator=",", max_line_width=99999) + "\n")
print("Next activity prediction evaluation finished.")
