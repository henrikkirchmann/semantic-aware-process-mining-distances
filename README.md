# Benchmarking Distributional Similarity between Activities in Event Data



This repository accompanies the ICPM 2025 submission by Kirchmann et al.:

**Let’s Simply Count: Quantifying Distributional Similarity between Activities in Event Data**  

We provide a comprehensive benchmarking framework to evaluate distributional similarity between activities in event data for process mining based on:
- **Intrinsic similarity quality**
- **Downstream performance** (Next Activity Prediction)
- **Computational efficiency**

---

## 📁 Scripts Overview

| Script                                                                                                                                    | Description |
|-------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| [`intrinsic_evaluation.py`](./evaluation/evaluation_of_activity_distances/intrinsic_evaluation/evaluation_activity_distance_intrinsic.py) | Assesses the quality of distributional similarity between activities using synthetic ground truth logs. |
| [`next_activity_prediction.py`](./evaluation/evaluation_of_activity_distances/next_activity_prediction)                                   | Benchmarks the embeddings in the task of next activity prediction. |
| [`runtime_analysis.py`](./evaluation/evaluation_of_activity_distances/runtime_analysis/runtime_analysis.py)                               | Analyzes runtime performance across methods. |


## Methods Compared in the Benchmark

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

| Method | Description                                         | Paper                                                                                                                                                                     | Original Implementation | File Location in this Repo |
|--------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|-----------------------------|
| **Substitution Scores** | Co-occurrence-based similarity using log-ratios     | Context Aware Trace Clustering: Towards Improving Process Mining Results; Bose & van der Aalst (2009), [DOI](https://doi.org/10.1137/1.9781611972795.35)                  | Re-implemented from authors' Java code | `distances/activity_distances/substitution_scores/` |
| **act2vec** | Neural embeddings adapted from Word2Vec (CBOW & Skip-gram) | act2vec, trace2vec, log2vec, and model2vec: Representation Learning for Business Processes; De Koninck et al. (2018), [DOI](https://doi.org/10.1007/978-3-319-98648-7_18) | [processmining.be/replearn](https://processmining.be/replearn/) | `models/act2vec/` |
| **Embedding Process Structure** | Feature vectors based on Petri nets and process structure | Embedding Process Structure in Activities for Process Mapping and Comparison; Chiorrini et al. (2022)  [DOI](https://doi.org/10.1007/978-3-031-15743-1_12)                | [GitHub Repo](https://github.com/KDMG/Embedding-Structure-in-Activities) | `models/structure_embedding/` |
| **Autoencoder** | Context-based representation learning via autoencoders | Learning Context-Based Representations of Events in Complex Processes; Gamallo-Fernandez et al. (2023), [DOI](https://doi.org/10.1109/ICWS60048.2023.00041)               | [GitLab Source](https://gitlab.citius.gal/pedro.gamallo/PM_Embeddings/-/blob/master/src/embeddings_generator/aerac.py?ref_type=heads) | `models/autoencoder/` |

---

#### 📄 Existing Methods – Implementation Details

- **Substitution Scores:**  
  Re-implemented in Python from the Java code shared by the authors.

- **act2vec:**  
  Used the default configurations provided by the authors from the official implementation.

- **Embedding Process Structure:**
 We re-implemented the computation of the path length, parallelism path length, and parallelism features.\
 The original implementation for the path lengths feature had a slow runtime that was not feasible for our intrinsic evaluation, so we optimized it, reducing computation time for some logs from several minutes to mere milliseconds. The original computation iterated over all transitions of a Petri net, and within each iteration, it further iterated over all paths leading to the transition. Within this second iteration, there was yet another loop that iterated over all nodes within each path.\
 In contrast, our implementation computes the path length feature using a bottom-up approach on the process tree, requiring only a single iteration over all nodes in the tree. \
 Additionally, the parallelism and parallelism path length features were initially only compatible with the authors' Petri net files from their GitHub repository.

- **Autoencoder:**  
  The original method used 5-fold cross-validation (64% train, 16% val, 20% test).  
  After completing all five folds, the embedding that performs best on the test fold is selected.\
  To reduce the considerable runtime of the full 5-fold procedure, we employed a simpler 80\%/20\% train/validation split. The authors do not specify an exact embedding dimension; instead, they match it to the requirements of the prediction model. Following their observation that larger embeddings improve performance, we fix the embedding dimension to 128. \
  Due to the long runtime, we tested the autoencoder only at a window size of 3, following the original study’s finding that this setting yields the best predictive performance.
---
