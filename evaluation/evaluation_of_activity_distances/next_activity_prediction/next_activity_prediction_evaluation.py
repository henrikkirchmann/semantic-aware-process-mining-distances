"""
Standalone Training and Next-Activity Prediction Script Using Pre-Split XES Files

This script loads a full XES log (from raw_datasets) and pre-split training, validation, and test
XES files (from split_datasets). It uses PM4Py to load the logs and converts each trace into a list of tokens.
Activity embeddings are computed from the validation set via one of our intrinsic methods.
Then the script vectorizes the traces (replacing one-hot with the computed embedding vectors),
builds an LSTM model, trains it (if enabled), and finally evaluates next-activity prediction.
Only next-activity prediction is performed here.

Author: Modified for standalone use
"""

import os, sys, time, copy, random
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
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

# --------------------------------------------------
# HARD-CODED PATHS FOR PRE-SPLIT XES FILES
# --------------------------------------------------
# Update these paths to where you downloaded the PredictiveMonitoringDatasets repo.
# Here we assume the raw full log is in "raw_datasets" and the pre-split files are in "split_datasets".
from definitions import ROOT_DIR

path_to_na = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "next_activity_prediction")

# For example, using BPI_Challenge_2012_A:
full_dataset = os.path.join(path_to_na, "raw_datasets", "BPI_Challenge_2012_A.xes.gz")

train_dataset = os.path.join(path_to_na, "split_datasets", "train_BPI_Challenge_2012_A.xes.gz")

val_dataset   = os.path.join(path_to_na, "split_datasets", "val_BPI_Challenge_2012_A.xes.gz")

test_dataset  = os.path.join(path_to_na, "split_datasets", "test_BPI_Challenge_2012_A.xes.gz")


# --------------------------------------------------
# DATA LOADING: Using PM4Py to load XES files and converting each trace to a list of tokens.
# --------------------------------------------------
from pm4py.objects.log.importer.xes import importer as xes_importer

def load_file_xes(file_path):
    """
    Loads an XES log and converts each trace into:
      - a list of activity tokens (ending with '!' as termination)
      - lists of time differences:
            times: time since previous event (in seconds)
            times2: time since start of trace
            times3: seconds since midnight for each event
            times4: weekday (0=Monday, …, 6=Sunday)
      - a list of case IDs
    """
    log = xes_importer.apply(file_path)
    lines = []       # List of traces; each trace is a list of tokens.
    timeseqs = []    # List of lists of time differences (since previous event)
    timeseqs2 = []   # List of lists: time since start of trace (in seconds)
    timeseqs3 = []   # List of lists: seconds since midnight for each event
    timeseqs4 = []   # List of lists: weekday for each event
    caseids = []
    for trace in log:
        # Try to get a case id, or generate one.
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
        # Append termination token '!'
        tokens.append('!')
        lines.append(tokens)
        timeseqs.append(times)
        timeseqs2.append(times2)
        timeseqs3.append(times3)
        timeseqs4.append(times4)
    return lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, caseids

# Load logs from pre-split files
print("Loading full log from:", full_dataset)
lines_full, ts_full, ts2_full, ts3_full, ts4_full, caseids_full = load_file_xes(full_dataset)
print("Loading training log from:", train_dataset)
lines_train, ts_train, ts2_train, ts3_train, ts4_train, caseids_train = load_file_xes(train_dataset)
print("Loading validation log from:", val_dataset)
lines_val, ts_val, ts2_val, ts3_val, ts4_val, caseids_val = load_file_xes(val_dataset)
print("Loading test log from:", test_dataset)
lines_test, ts_test, ts2_test, ts3_test, ts4_test, caseids_test = load_file_xes(test_dataset)

# Compute maximum trace length over the full log
maxlen = max(len(trace) for trace in lines_full)
print("Maximum trace length:", maxlen)

# Compute time divisors based only on the training set (for normalization)
all_times = [t for trace in ts_train for t in trace]
divisor = np.mean(all_times) if all_times else 1
all_times2 = [t for trace in ts2_train for t in trace]
divisor2 = np.mean(all_times2) if all_times2 else 1
print("Time divisor (avg time diff):", divisor)
print("Time divisor2 (avg time since start):", divisor2)

# Build the vocabulary for target activities from the full log.
vocab = sorted(set(token for trace in lines_full for token in trace))
# We need to keep the termination token for target encoding.
target_tokens = copy.copy(vocab)
if '!' in vocab:
    vocab.remove('!')
target_token_indices = {token: i for i, token in enumerate(target_tokens)}
indices_token = {i: token for i, token in enumerate(target_tokens)}

# --------------------------------------------------
# EMBEDDING EXTRACTION FUNCTIONS
# --------------------------------------------------
# We use our two methods (from our project files) to compute activity embeddings.

from distances.activity_distances.activity_context_frequency.activity_contex_frequency import get_activity_context_frequency_matrix
from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import get_activity_activity_co_occurence_matrix

def get_activity_embeddings_from_val_context_frequency(lines_val, ngram_size=3):
    # Prepare the log as a list of lists (without termination token)
    log_val = [trace[:-1] for trace in lines_val]  # remove '!'
    alphabet_val = sorted(set(token for trace in log_val for token in trace))
    # get_activity_context_frequency_matrix returns: distance_matrix, embeddings, ...
    _, embeddings, _, _, _ = get_activity_context_frequency_matrix(log_val, alphabet_val, ngram_size, bag_of_words=2)
    return embeddings

def get_activity_embeddings_from_val_activity_cooccurrence(lines_val, ngram_size=3, bag_of_words=False):
    log_val = [trace[:-1] for trace in lines_val]
    alphabet_val = sorted(set(token for trace in log_val for token in trace))
    _, embeddings, _, _ = get_activity_activity_co_occurence_matrix(log_val, alphabet_val, ngram_size, bag_of_words)
    return embeddings

# Select which embedding method to use (hard-coded)
selected_embedding_method = "activity_cooccurrence"
if selected_embedding_method == "context_frequency":
    activity_embeddings = get_activity_embeddings_from_val_context_frequency(lines_val)
elif selected_embedding_method == "activity_cooccurrence":
    activity_embeddings = get_activity_embeddings_from_val_activity_cooccurrence(lines_val)
else:
    raise ValueError("Unknown embedding method")
embedding_dim = len(next(iter(activity_embeddings.values())))
num_features = embedding_dim + 5
print("Using embedding method:", selected_embedding_method, "with embedding dimension:", embedding_dim)

# --------------------------------------------------
# VECTORIZATION FUNCTION (for preprocessed traces)
# --------------------------------------------------
def vectorize_fold(lines_fold, ts_fold, ts2_fold, ts3_fold, ts4_fold, divisor, divisor2, activity_embeddings, embedding_dim):
    """
    For each trace (a list of tokens), create multiple prefix sequences.
    Each sequence is left-padded to maxlen and includes the embedding vector for the activity.
    Also constructs one-hot encoded target vectors.
    """
    sentences = []
    next_tokens = []
    sent_ts = []
    sent_ts2 = []
    sent_ts3 = []
    sent_ts4 = []
    # Create prefixes from each trace (skip empty prefix)
    for tokens, t_seq, t2_seq, t3_seq, t4_seq in zip(lines_fold, ts_fold, ts2_fold, ts3_fold, ts4_fold):
        for i in range(1, len(tokens)):
            sentences.append(tokens[:i])
            sent_ts.append(t_seq[:i])
            sent_ts2.append(t2_seq[:i])
            sent_ts3.append(t3_seq[:i])
            sent_ts4.append(t4_seq[:i])
            next_tokens.append(tokens[i])
    print("Number of sequences:", len(sentences))
    X = np.zeros((len(sentences), maxlen, embedding_dim + 5), dtype=np.float32)
    y_act = np.zeros((len(sentences), len(target_tokens)), dtype=np.float32)
    y_time = np.zeros((len(sentences)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen - len(sentence)
        for t, token in enumerate(sentence):
            # Look up the embedding vector (if not found, use zeros)
            if token in activity_embeddings:
                X[i, t + leftpad, :embedding_dim] = activity_embeddings[token]
            else:
                X[i, t + leftpad, :embedding_dim] = np.zeros(embedding_dim)
            # Additional features:
            X[i, t + leftpad, embedding_dim]     = t + 1
            X[i, t + leftpad, embedding_dim + 1] = sent_ts[i][t] / divisor
            X[i, t + leftpad, embedding_dim + 2] = sent_ts2[i][t] / divisor2
            X[i, t + leftpad, embedding_dim + 3] = sent_ts3[i][t] / 86400
            X[i, t + leftpad, embedding_dim + 4] = sent_ts4[i][t] / 7
        # Build one-hot target vector for next token
        target = next_tokens[i]
        for token in target_tokens:
            if token == target:
                y_act[i, target_token_indices[token]] = 1
        # For time prediction target, use the next time diff (normalized)
        y_time[i] = sent_ts[i][-1] / divisor
    return X, target_token_indices, y_act, y_time

# Vectorize each fold using our new function:
X_full, _, y_act_full, y_time_full = vectorize_fold(lines_full, ts_full, ts2_full, ts3_full, ts4_full, divisor, divisor2, activity_embeddings, embedding_dim)
X_train, _, y_act_train, y_time_train = vectorize_fold(lines_train, ts_train, ts2_train, ts3_train, ts4_train, divisor, divisor2, activity_embeddings, embedding_dim)
X_val, _, y_act_val, y_time_val = vectorize_fold(lines_val, ts_val, ts2_val, ts3_val, ts4_val, divisor, divisor2, activity_embeddings, embedding_dim)
X_test, _, y_act_test, y_time_test = vectorize_fold(lines_test, ts_test, ts2_test, ts3_test, ts4_test, divisor, divisor2, activity_embeddings, embedding_dim)

# --------------------------------------------------
# BUILD THE MODEL
# --------------------------------------------------
print("Building model...")
main_input = Input(shape=(maxlen, num_features), name='main_input')
l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input)
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(len(target_tokens), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
model = Model(inputs=[main_input], outputs=[act_output, time_output])
opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
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
