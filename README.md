# Benchmarking Distributional Similarity between Activities in Event Data



This repository accompanies the ICPM 2025 submission by Kirchmann et al.:

**Let’s Simply Count: Quantifying Distributional Similarity between Activities in Event Data**  


---

## 📁 Benchmark Overview
We provide a comprehensive benchmarking framework to evaluate distributional similarity between activities in event data for process mining based on:

| Benchmark                                               | Description |
|---------------------------------------------------------|-------------|
| [Intrinsic Evaluation](#🧪-intrinsic-evaluation)        | Assesses the quality of distributional similarity between activities using synthetic ground truth logs. |
| [Next Activity Prediction](#⏩-next-activity-prediction) | Benchmarks the embeddings in the task of next activity prediction. |
| [Runtime Analysis](#⏳-runtime-analysis)                 | Analyzes runtime performance across methods. |


## ⚙️ Methods Compared in the Benchmark

The benchmarking framework evaluates both **newly proposed methods** and **existing methods** for computing distributional similarity between activities in event data.

The implementation for each method can be found in [`distances/activity_distances folder`](./distances/activity_distances).


### 🆕 New Count-based Methods

Our proposed methods derive activity embeddings and distributional similarity from event logs using simple and interpretable **count-based approaches**. Each variant is defined by its matrix type, context interpretation, and optional postprocessing.

**📁 Where to find them:**

- **Activity-activity co-occurrence matrices:**  
  [`distances/activity_distances/activity_activity_co_occurence/activity_activity_co_occurrence.py`](./distances/activity_distances/activity_activity_co_occurence/activity_activity_co_occurrence.py)
- **Activity-context frequency matrices:**  
  [`distances/activity_distances/activity_context_frequency/activity_contex_frequency.py`](./distances/activity_distances/activity_context_frequency/activity_contex_frequency.py)
- **PMI & PPMI postprocessing:**  
  [`distances/activity_distances/pmi/pmi.py`](./distances/activity_distances/pmi/pmi.py)  



###  📖 Existing Methods

These baselines are implemented or re-used from prior work.

| Method | Description                                         | Paper                                                                                                                                                                     | Original Implementation | File Location in this Repo                                                                                                                                                                                                                           |
|--------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Substitution Scores** | Co-occurrence-based similarity using log-ratios     | Context Aware Trace Clustering: Towards Improving Process Mining Results; Bose & van der Aalst (2009), [DOI](https://doi.org/10.1137/1.9781611972795.35)                  | Re-implemented from authors' Java code | [`distances/activity_distances/bose_2009_context_aware_trace_clustering/algorithm.py`](./distances/activity_distances/bose_2009_context_aware_trace_clustering/algorithm.py)                                                                         |
| **act2vec** | Neural embeddings adapted from Word2Vec (CBOW & Skip-gram) | act2vec, trace2vec, log2vec, and model2vec: Representation Learning for Business Processes; De Koninck et al. (2018), [DOI](https://doi.org/10.1007/978-3-319-98648-7_18) | [processmining.be/replearn](https://processmining.be/replearn/) | [`distances/activity_distances/de_koninck_2018_act2vec/algorithm.py`](./distances/activity_distances/de_koninck_2018_act2vec/algorithm.py)                                                                                                           |
| **Embedding Process Structure** | Feature vectors based on Petri nets and process structure | Embedding Process Structure in Activities for Process Mapping and Comparison; Chiorrini et al. (2022)  [DOI](https://doi.org/10.1007/978-3-031-15743-1_12)                | [GitHub Repo](https://github.com/KDMG/Embedding-Structure-in-Activities) | [`distances/activity_distances/chiorrini_2022_embedding_process_structure/embedding_process_structure.py`](./distances/activity_distances/chiorrini_2022_embedding_process_structure/embedding_process_structure.py)                                 |
| **Autoencoder** | Context-based representation learning via autoencoders | Learning Context-Based Representations of Events in Complex Processes; Gamallo-Fernandez et al. (2023), [DOI](https://doi.org/10.1109/ICWS60048.2023.00041)               | [GitLab Source](https://gitlab.citius.gal/pedro.gamallo/PM_Embeddings/-/blob/master/src/embeddings_generator/aerac.py?ref_type=heads) | [`distances/activity_distances/gamallo_fernandez_2023_context_based_representations/src/embeddings_generator/main_new.py`](./distances/activity_distances/gamallo_fernandez_2023_context_based_representations/src/embeddings_generator/main_new.py) |

---

#### 📄 Existing Methods – Implementation Details

- **Substitution Scores:**\
  Re-implemented in Python from the Java code shared by the authors.

- **act2vec:**\
  Used the default configurations provided by the authors from the official implementation.

- **Embedding Process Structure:**\
 We re-implemented the computation of the path length, parallelism path length, and parallelism features.\
 The original implementation for the path lengths feature had a slow runtime that was not feasible for our intrinsic evaluation, so we optimized it, reducing computation time for some logs from several minutes to mere milliseconds. The original computation iterated over all transitions of a Petri net, and within each iteration, it further iterated over all paths leading to the transition. Within this second iteration, there was yet another loop that iterated over all nodes within each path.\
 In contrast, our implementation computes the path length feature using a bottom-up approach on the process tree, requiring only a single iteration over all nodes in the tree. \
 Additionally, the parallelism and parallelism path length features were initially only compatible with the authors' Petri net files from their GitHub repository.

- **Autoencoder:**\
  The original method used 5-fold cross-validation (64% train, 16% val, 20% test).  
  After completing all five folds, the embedding that performs best on the test fold is selected.\
  To reduce the considerable runtime of the full 5-fold procedure, we employed a simpler 80\%/20\% train/validation split. The authors do not specify an exact embedding dimension; instead, they match it to the requirements of the prediction model. Following their observation that larger embeddings improve performance, we fix the embedding dimension to 128. \
  Due to the long runtime, we tested the autoencoder only at a window size of 3, following the original study’s finding that this setting yields the best predictive performance.
---

## 🧪 Intrinsic Evaluation

### 🔧 How to Run the Script

To execute the benchmark evaluation, simply run the script [`evaluation/evaluation_of_activity_distances/intrinsic_evaluation/intrinsic_evaluation.py`](./evaluation/evaluation_of_activity_distances/intrinsic_evaluation/intrinsic_evaluation.py). 

---

### ⚙️ Configurable Settings

You can customize the experiment by editing the `if __name__ == '__main__':` section of the script. Here's a breakdown of what you can change:

#### Similarity / Embedding Methods
These define which methods will be evaluated. They are appended to `activity_distance_functions`. You can **enable/disable methods** by commenting/uncommenting `append()` lines.

Example:
```python
# activity_distance_functions.append("Unit Distance")  # Not used
activity_distance_functions.append("Bose 2009 Substitution Scores")  # Used
```

You can also adjust the **window sizes** that are evaluated via:
```python
window_size_list = [3, 5, 9]
```

---

####  Parameters for Ground Truth Log Creation
These control the synthetic event log creation: For each original log, we consider all combinations of
how many differnt activities are replaced $\{1, 2, ..., \min(|A_L|, r\_min)\}$ and with how many new activites are each original activty with $\{2, 3, ..., w\}$ 
```python
r_min         = 10     # How many differnt activities are replaced  
w             = 5      # How many new activites are used to replace each original activty with
sampling_size = 5      # How many samples to draw
create_new    = True   # True = generate new logs; False = use pre-generated ones
```

> 💡 If `create_new = False`, make sure to unzip and place the folders of the pre-generated logs from  
> [https://box.hu-berlin.de/d/7a97101239654eae8e6c/](https://box.hu-berlin.de/d/7a97101239654eae8e6c/) into:  
> [`evaluation/evaluation_of_activity_distances/intrinsic_evaluation/newly_created_logs/`](evaluation/evaluation_of_activity_distances/intrinsic_evaluation/newly_created_logs/)

---

#### Event Logs to Evaluate

By default, only the `'Sepsis'` log is evaluated:
```python
log_list.append('Sepsis')
```

To evaluate more logs, uncomment the entries in the multi-line string:
```python
"""
log_list.append('BPIC12')
log_list.append('BPIC13_incidents')
...
"""
```
---

You can download the original event logs from the following location:

[https://box.hu-berlin.de/f/aa5905ab235e444b8ffa/](https://box.hu-berlin.de/f/aa5905ab235e444b8ffa/)

Unpack the folder and place all `.xes.gz` files into the directory: [`event_logs`](./event_logs) 

### Results


## ⏩ Next Activity Prediction

## ⏳ Runtime Analysis

## 🗃️ Datasets


### Event Log Statistics

| Event Log                                                                                                                                     |   # Unique Activities |   # of Traces |   #Variants/#Traces | Min. Trace Length |   Avg. TL |   Max. TL | Intrinsic   | Next Act.   | Runtime   |
|:----------------------------------------------------------------------------------------------------------------------------------------------|----------------------:|--------------:|--------------------:|------------------:|----------:|----------:|:------------|:------------|:----------|
| [BPIC12](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                          |                    24 |         13087 |                0.33 |                 3 |     20.04 |       175 | ✓           | ×           | ×         |
| [BPIC12 A](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                        |                    10 |         13087 |                0    |                 3 |      4.65 |         8 | ✓           | ✓           | ×         |
| [BPIC12 C.](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                       |                    23 |         13087 |                0.33 |                 3 |     12.57 |        96 | ✓           | ×           | ×         |
| [BPIC12 O](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                        |                     7 |          5015 |                0.03 |                 3 |      6.23 |        30 | ✓           | ✓           | ×         |
| [BPIC12 W](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                        |                     7 |          9658 |                0.27 |                 2 |     17.61 |       156 | ✓           | ✓           | ×         |
| [BPIC12 W C.](https://data.4tu.nl/collections/BPI_Challenge_2012/5065419)                                                                     |                     6 |          9658 |                0.23 |                 1 |      7.5  |        74 | ✓           | ×           | ×         |
| [BPIC13 C. P.](https://data.4tu.nl/collections/_/5065448/1)                                                                             |                     7 |          1487 |                0.22 |                 1 |      4.48 |        35 | ✓           | ✓           | ×         |
| [BPIC13 I.](https://data.4tu.nl/collections/_/5065448/1)                                                                                |                     4 |          7554 |                0.2  |                 1 |      8.68 |       123 | ✓           | ✓           | ×         |
| [BPIC13 O. P.](https://data.4tu.nl/collections/_/5065448/1)                                                                             |                     3 |           819 |                0.13 |                 1 |      2.87 |        22 | ✓           | ✓           | ×         |
| [BPIC15 1](https://data.4tu.nl/collections/_/5065424/1)                                                                                 |                   398 |          1199 |                0.98 |                 2 |     43.55 |       101 | ✓           | ✓           | ✓         |
| [BPIC15 2](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   410 |           832 |                1    |                 1 |     53.31 |       132 | ✓           | ✓           | ×         |
| [BPIC15 3](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   383 |          1409 |                0.96 |                 3 |     42.36 |       124 | ✓           | ✓           | ×         |
| [BPIC15 4](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   356 |          1053 |                1    |                 1 |     44.91 |       116 | ✓           | ✓           | ×         |
| [BPIC15 5](https://data.4tu.nl/collections/_/5065424/1)                                                                                       |                   389 |          1156 |                1    |                 5 |     51.11 |       154 | ✓           | ✓           | ×         |
| [BPIC17](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884/1)                                                                  |                    26 |         31509 |                0.51 |                10 |     38.16 |       180 | ✓           | ×           | ×         |
| [BPIC18](https://data.4tu.nl/articles/dataset/BPI_Challenge_2018/12688355/1)                                                                  |                    41 |         43809 |                0.65 |                24 |     57.39 |      2973 | ✓           | ×           | ×         |
| [BPIC19](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853/1)                                                                  |                    42 |        251734 |                0.05 |                 1 |      6.34 |       990 | ✓           | ×           | ✓         |
| [BPIC20 D. D.](https://data.4tu.nl/datasets/6a0a26d2-82d0-4018-b1cd-89afb0e8627f)                                                                  |                    17 |         10500 |                0.01 |                 1 |      5.37 |        24 | ✓           | ✓           | ×         |
| [BPIC20 I. D.](https://data.4tu.nl/datasets/91fd1fa8-4df4-4b1a-9a3f-0116c412378f)                                                                  |                    34 |          6449 |                0.12 |                 3 |     11.19 |        27 | ✓           | ✓           | ×         |
| [BPIC20 P. L.](https://data.4tu.nl/datasets/db35afac-2133-40f3-a565-2dc77a9329a3)                                                                  |                    51 |          7065 |                0.21 |                 3 |     12.25 |        90 | ✓           | ✓           | ×         |
| [BPIC20 P. T. C.](https://data.4tu.nl/datasets/fb84cf2d-166f-4de2-87be-62ee317077e5)                                                               |                    29 |          2099 |                0.1  |                 1 |      8.69 |        21 | ✓           | ✓           | ×         |
| [BPIC20 R. F. P.](https://data.4tu.nl/datasets/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc)                                                               |                    19 |          6886 |                0.01 |                 1 |      5.34 |        20 | ✓           | ✓           | ×         |
| [CCC19](https://data.4tu.nl/articles/_/12714932/1)                                                                                            |                    29 |            20 |                1    |                52 |     69.7  |       118 | ✓           | ×           | ×         |
| [Env Permit](https://data.4tu.nl/articles/dataset/Receipt_phase_of_an_environmental_permit_application_process_WABO_CoSeLoG_project/12709127) |                    27 |          1434 |                0.08 |                 1 |      5.98 |        25 | ✓           | ✓           | ×         |
| [Helpdesk](https://data.4tu.nl/articles/_/12675977/1)                                                                                         |                    14 |          4580 |                0.05 |                 2 |      4.66 |        15 | ✓           | ✓           | ×         |
| [Hospital Billing](https://data.4tu.nl/articles/_/12705113/1)                                                                                 |                    18 |        100000 |                0.01 |                 1 |      4.51 |       217 | ✓           | ×           | ×         |
| [Nasa](https://data.4tu.nl/articles/_/12696995/1)                                                                                             |                    47 |          2566 |                0.98 |                12 |     28.7  |        50 | ✓           | ✓           | ×         |
| [RTFM](https://data.4tu.nl/articles/_/12683249/1)                                                                                             |                    11 |        150370 |                0    |                 2 |      3.73 |        20 | ✓           | ×           | ×         |
| [Sepsis](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639/1)                                                            |                    16 |          1050 |                0.81 |                 3 |     14.49 |       185 | ✓           | ✓           | ×         |
