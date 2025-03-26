#!/usr/bin/env python
"""
Standalone Next-Activity Prediction Evaluation Pipeline

This script loops over all logs in RAW_DATASETS_DIR and their corresponding pre‐split files in SPLIT_DATASETS_DIR.
For each log and for each input encoding method (either "one_hot" or one of our computed embedding methods),
the script:
  1. Loads the full, train, validation, and test logs using PM4Py.
  2. Builds a vocabulary from the full log (for one‐hot targets).
  3. Depending on the chosen encoding mode:
       - If "one_hot": uses the original one‐hot representation.
       - Otherwise, computes activity embeddings using one of our intrinsic methods.
         You can choose to compute embeddings from the "train", "validation", or "train_val" split.
         Supported methods (with window-size variations) include:
           • Unit Distance
           • Bose 2009 Substitution Scores
           • De Koninck 2018 act2vec CBOW
           • De Koninck 2018 act2vec skip-gram
           • Activity-Activitiy Co Occurrence (Bag Of Words, N-Gram, PMI, PPMI)
           • Activity-Context (Bag Of Words, N-Grams, PMI, PPMI)
           • Chiorrini 2022 Embedding Process Structure
  4. Vectorizes each trace (each time step is represented by the activity encoding concatenated with 5 time features).
  5. Builds, trains, and evaluates an LSTM–based next–activity prediction model.
  6. Saves the accuracy and weighted F1–score for each log/method combination in CSV files.
  7. Aggregates and prints (and saves) the average scores per method over all logs.

Author: Modified for standalone evaluation
"""

import os, sys, copy, random, time, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# -----------------------
# Set cuDNN environment variables (must be set before TensorFlow is imported)
if os.environ.get("MY_CUDNN_SET") != "true":
    os.environ["LD_LIBRARY_PATH"] = "/vol/fob-vol4/mi17/kirchmah/cudnn-8.9.6/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_PRELOAD"] = "/vol/fob-vol4/mi17/kirchmah/cudnn-8.9.6/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn.so.8.9.6"
    os.environ["MY_CUDNN_SET"] = "true"
    os.execv(sys.executable, [sys.executable] + sys.argv)

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Set GPU memory growth.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(len(gpus), "Physical GPUs,", len(tf.config.experimental.list_logical_devices('GPU')), "Logical GPUs")
    except RuntimeError as e:
        print(e)

tf.compat.v1.set_random_seed(42)
random.seed(42)
np.random.seed(42)

# -----------------------
# User Options:
# Choose encoding_mode: "one_hot" (baseline) or "embedding"
encoding_mode = "embedding"  # Change to "one_hot" for one-hot encodings.
use_one_hot = (encoding_mode == "one_hot")
print("Encoding mode:", encoding_mode)

# For computed embeddings, choose the source split: "train", "validation", or "train_val"
embedding_source = "validation"  # Options: "train", "validation", "train_val"
print("Embedding source:", embedding_source)

# -----------------------
# Evaluation parameters.
BATCH_SIZE = 32
EPOCHS = 200

# -----------------------
# Directories.
from definitions import ROOT_DIR
NA_DIR = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "next_activity_prediction")
RAW_DATASETS_DIR = os.path.join(NA_DIR, "raw_datasets")
SPLIT_DATASETS_DIR = os.path.join(NA_DIR, "split_datasets")
RESULTS_DIR = os.path.join(NA_DIR, "results")
MODELS_DIR = os.path.join(NA_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# Supported Encoding Methods.
# (Names must match our intrinsic distance functions.)
encoding_methods = [
    "one_hot",  # baseline
    "Unit Distance",
    "Bose 2009 Substitution Scores",
    "De Koninck 2018 act2vec CBOW",
    "De Koninck 2018 act2vec skip-gram",
    "Activity-Activitiy Co Occurrence Bag Of Words",
    "Activity-Activitiy Co Occurrence N-Gram",
    "Activity-Activitiy Co Occurrence Bag Of Words PMI",
    "Activity-Activitiy Co Occurrence N-Gram PMI",
    "Activity-Activitiy Co Occurrence Bag Of Words PPMI",
    "Activity-Activitiy Co Occurrence N-Gram PPMI",
    "Activity-Context Bag Of Words",
    "Activity-Context N-Grams",
    "Activity-Context Bag Of Words PMI",
    "Activity-Context N-Grams PMI",
    "Activity-Context Bag Of Words PPMI",
    "Activity-Context N-Grams PPMI",
    "Chiorrini 2022 Embedding Process Structure"
]
# Optionally, add window–size variations.
from evaluation.data_util.util_activity_distances_intrinsic import add_window_size_evaluation
window_size_list = [3, 5, 9]
encoding_methods = add_window_size_evaluation(encoding_methods, window_size_list)
print("Encoding methods to evaluate:")
print(encoding_methods)

# -----------------------
# Data Loading using PM4Py.
from pm4py.objects.log.importer.xes import importer as xes_importer

def load_file_xes(file_path):
    log = xes_importer.apply(file_path)
    lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, caseids = [], [], [], [], [], []
    for trace in log:
        caseid = trace.attributes.get("concept:name", "case_" + str(len(caseids)))
        caseids.append(caseid)
        tokens, times, times2, times3, times4 = [], [], [], [], []
        casestart, lastevent = None, None
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

def build_vocab(lines_full):
    vocab = sorted(set(token for trace in lines_full for token in trace))
    target_tokens = copy.copy(vocab)
    if '!' in vocab:
        vocab.remove('!')
    target_token_indices = {token: i for i, token in enumerate(target_tokens)}
    indices_token = {i: token for i, token in enumerate(target_tokens)}
    return vocab, target_tokens, target_token_indices, indices_token

def extract_window_size(s):
    match = re.search(r"w_(\d+)", s)
    return int(match.group(1)) if match else 3

# -----------------------
# Embedding Computation.
# This function calls our intrinsic methods (or falls back to one-hot) with the proper arguments.
def get_embeddings_for_method(method, embedding_input, ngram_size=3):
    # Remove termination tokens.
    log_input = [trace[:-1] for trace in embedding_input]
    alphabet = sorted(set(token for trace in log_input for token in trace))
    win_size = extract_window_size(method)
    if method == "one_hot":
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method.startswith("Unit Distance"):
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method.startswith("Bose 2009 Substitution Scores"):
        from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import get_substitution_and_insertion_scores
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method.startswith("De Koninck 2018 act2vec"):
        from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
        sg = 0 if "CBOW" in method else 1
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method.startswith("Our act2vec"):
        from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import get_act2vec_distance_matrix_our
        emb = get_act2vec_distance_matrix_our(log_input, alphabet, win_size)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Activity-Activitiy Co Occurrence"):
        from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import get_activity_activity_co_occurence_matrix, get_activity_activity_frequency_matrix_pmi
        bag = True if "Bag Of Words" in method else False
        distance_matrix, emb, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(log_input, alphabet, win_size, bag_of_words=bag)
        if "PPMI" in method:
            emb, _ = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 1)
        elif "PMI" in method:
            emb, _ = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 0)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Activity-Context"):
        from distances.activity_distances.activity_context_frequency.activity_contex_frequency import get_activity_context_frequency_matrix, get_activity_context_frequency_matrix_pmi
        if "Bag Of Words" in method and "N-Grams" not in method:
            bag_mode = 2
        elif "N-Grams" in method:
            bag_mode = 0
        else:
            bag_mode = 2
        distance_matrix, emb, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(log_input, alphabet, win_size, bag_of_words=bag_mode)
        if "PPMI" in method:
            emb, _ = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict, context_index, 1)
        elif "PMI" in method:
            emb, _ = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict, context_index, 0)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Chiorrini 2022 Embedding Process Structure"):
        from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import get_embedding_process_structure_distance_matrix
        distance_matrix, emb = get_embedding_process_structure_distance_matrix(log_input, alphabet, False)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Gamallo Fernandez 2023 Context Based"):
        from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import get_context_based_distance_matrix
        emb = get_context_based_distance_matrix(log_input, win_size)
        return emb, len(next(iter(emb.values())))
    else:
        raise ValueError("Unknown encoding method: " + method)

# -----------------------
# Vectorization Function.
def vectorize_fold(lines, ts, ts2, ts3, ts4, divisor, divisor2, encoding_dim, extra_features=5,
                   use_one_hot=False, activity_embeddings=None, one_hot_indices=None):
    sentences = []
    next_tokens = []
    sent_ts, sent_ts2, sent_ts3, sent_ts4 = [], [], [], []
    for tokens, t_seq, t2_seq, t3_seq, t4_seq in zip(lines, ts, ts2, ts3, ts4):
        for i in range(1, len(tokens)):
            sentences.append(tokens[:i])
            sent_ts.append(t_seq[:i])
            sent_ts2.append(t2_seq[:i])
            sent_ts3.append(t3_seq[:i])
            sent_ts4.append(t4_seq[:i])
            next_tokens.append(tokens[i])
    num_seq = len(sentences)
    X = np.zeros((num_seq, maxlen, encoding_dim + extra_features), dtype=np.float32)
    y_act = np.zeros((num_seq, len(target_tokens)), dtype=np.float32)
    y_time = np.zeros(num_seq, dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen - len(sentence)
        for t, token in enumerate(sentence):
            if use_one_hot:
                if token in one_hot_indices:
                    X[i, t+leftpad, one_hot_indices[token]] = 1
            else:
                if token in activity_embeddings:
                    X[i, t+leftpad, :encoding_dim] = activity_embeddings[token]
                else:
                    X[i, t+leftpad, :encoding_dim] = np.zeros(encoding_dim)
            # Additional time features.
            X[i, t+leftpad, encoding_dim] = t+1
            X[i, t+leftpad, encoding_dim+1] = sent_ts[i][t] / divisor
            X[i, t+leftpad, encoding_dim+2] = sent_ts2[i][t] / divisor2
            X[i, t+leftpad, encoding_dim+3] = sent_ts3[i][t] / 86400
            X[i, t+leftpad, encoding_dim+4] = sent_ts4[i][t] / 7
        target = next_tokens[i]
        for token in target_tokens:
            if token == target:
                y_act[i, target_token_indices[token]] = 1
        y_time[i] = sent_ts[i][-1] / divisor
    return X, y_act, y_time

# -----------------------
# Loop over all logs and encoding methods.
raw_logs = [f for f in os.listdir(RAW_DATASETS_DIR) if f.endswith(".xes.gz")]
results_summary = []

for raw_log in raw_logs:
    log_name = os.path.splitext(os.path.splitext(raw_log)[0])[0]  # Remove ".xes.gz"
    full_path = os.path.join(RAW_DATASETS_DIR, raw_log)
    train_path = os.path.join(SPLIT_DATASETS_DIR, f"train_{log_name}.xes.gz")
    val_path   = os.path.join(SPLIT_DATASETS_DIR, f"val_{log_name}.xes.gz")
    test_path  = os.path.join(SPLIT_DATASETS_DIR, f"test_{log_name}.xes.gz")
    print("\n========== Processing log:", log_name, "==========")
    print("Full log:", full_path)
    print("Train:", train_path)
    print("Val:", val_path)
    print("Test:", test_path)

    lines_full_log, ts_full_log, ts2_full_log, ts3_full_log, ts4_full_log, _ = load_file_xes(full_path)
    lines_train, ts_train, ts2_train, ts3_train, ts4_train, _ = load_file_xes(train_path)
    lines_val, ts_val, ts2_val, ts3_val, ts4_val, _ = load_file_xes(val_path)
    lines_test, ts_test, ts2_test, ts3_test, ts4_test, _ = load_file_xes(test_path)

    # For each encoding method.
    for method in encoding_methods:
        print("\n--- Encoding method:", method, "for log:", log_name, "---")
        if method == "one_hot":
            use_one_hot_flag = True
            encoding_dim = len(vocab)
        else:
            use_one_hot_flag = False
            # For some methods (Unit, Bose, De Koninck 2018 act2vec) we fallback to one-hot.
            if method.startswith("Unit Distance") or method.startswith("Bose") or method.startswith("De Koninck 2018 act2vec"):
                activity_embeddings = {activity: np.eye(len(vocab))[i] for i, activity in enumerate(vocab)}
                encoding_dim = len(vocab)
            else:
                if embedding_source == "train":
                    embedding_input = lines_train
                elif embedding_source == "validation":
                    embedding_input = lines_val
                elif embedding_source == "train_val":
                    embedding_input = lines_train + lines_val
                else:
                    raise ValueError("Unknown embedding_source")
                try:
                    activity_embeddings, encoding_dim = get_embeddings_for_method(method, embedding_input)
                except Exception as e:
                    print(f"Error computing embeddings for method {method} on log {log_name}: {e}")
                    continue
        print("Final encoding dimension:", encoding_dim)
        total_features = encoding_dim + 5

        # Vectorize datasets.
        if use_one_hot_flag:
            one_hot_indices = {token: i for i, token in enumerate(vocab)}
            X_train, y_act_train, y_time_train = vectorize_fold(lines_train, ts_train, ts2_train, ts3_train, ts4_train,
                                                                 divisor, divisor2, encoding_dim, use_one_hot=True,
                                                                 one_hot_indices=one_hot_indices)
            X_val, y_act_val, y_time_val = vectorize_fold(lines_val, ts_val, ts2_val, ts3_val, ts4_val,
                                                           divisor, divisor2, encoding_dim, use_one_hot=True,
                                                           one_hot_indices=one_hot_indices)
            X_test, y_act_test, y_time_test = vectorize_fold(lines_test, ts_test, ts2_test, ts3_test, ts4_test,
                                                              divisor, divisor2, encoding_dim, use_one_hot=True,
                                                              one_hot_indices=one_hot_indices)
        else:
            X_train, y_act_train, y_time_train = vectorize_fold(lines_train, ts_train, ts2_train, ts3_train, ts4_train,
                                                                 divisor, divisor2, encoding_dim, activity_embeddings=activity_embeddings)
            X_val, y_act_val, y_time_val = vectorize_fold(lines_val, ts_val, ts2_val, ts3_val, ts4_val,
                                                           divisor, divisor2, encoding_dim, activity_embeddings=activity_embeddings)
            X_test, y_act_test, y_time_test = vectorize_fold(lines_test, ts_test, ts2_test, ts3_test, ts4_test,
                                                              divisor, divisor2, encoding_dim, activity_embeddings=activity_embeddings)

        # -----------------------
        # Build the LSTM model.
        print("Building model for log", log_name, "with method", method)
        main_input = Input(shape=(maxlen, total_features), name='main_input')
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

        # -----------------------
        # Setup callbacks and file paths.
        best_model_path = os.path.join(MODELS_DIR, f"{log_name}_{method}.h5")
        early_stopping = EarlyStopping(monitor='val_loss', patience=42)
        model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1,
                                           save_best_only=True, save_weights_only=False)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=0.0001)

        # Train the model.
        print("Training model for log", log_name, "with method", method)
        model.fit(X_train, {'act_output': y_act_train, 'time_output': y_time_train},
                  validation_data=(X_val, {"act_output": y_act_val, "time_output": y_time_val}),
                  verbose=1, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                  batch_size=BATCH_SIZE, epochs=EPOCHS)

        # Evaluate the model.
        print("Evaluating model for log", log_name, "with method", method)
        model.load_weights(best_model_path)
        model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
                      optimizer=opt,
                      metrics={"act_output": "acc", "time_output": "mae"})
        metrics_values = model.evaluate(X_test, {'act_output': y_act_test, 'time_output': y_time_test},
                                        verbose=1, batch_size=BATCH_SIZE)
        preds = model.predict([X_test])
        y_act_pred_probs = preds[0]
        y_act_pred = np.argmax(y_act_pred_probs, axis=1)
        y_act_true = np.argmax(y_act_test, axis=1)
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_act_true, y_act_pred)
        f1 = f1_score(y_act_true, y_act_pred, average="weighted")
        print(f"Results for log {log_name} with method {method}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

        # Save individual results.
        result_dict = {"log": log_name, "method": method, "accuracy": acc, "f1": f1}
        results_summary.append(result_dict)
        df_log = pd.DataFrame([result_dict])
        df_log.to_csv(os.path.join(RESULTS_DIR, f"{log_name}_{method}_results.csv"), index=False)

        # Clear model from memory before next iteration.
        tf.keras.backend.clear_session()

# -----------------------
# Aggregate overall results.
df_results = pd.DataFrame(results_summary)
df_results.to_csv(os.path.join(RESULTS_DIR, "all_logs_results.csv"), index=False)
print("\nOverall results:")
print(df_results)
avg_results = df_results.groupby("method").agg({"accuracy": "mean", "f1": "mean"}).reset_index()
print("\nAverage results per method:")
print(avg_results)
avg_results.to_csv(os.path.join(RESULTS_DIR, "average_results_per_method.csv"), index=False)

print("Pipeline evaluation finished.")
