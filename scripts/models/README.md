# SCOTUS AI: Predictive Modeling of Judicial Votes

This directory contains the core machine learning components of the SCOTUS AI project. It implements a sophisticated neural network architecture designed to predict the voting outcomes of Supreme Court cases by deeply analyzing the interplay between the case's legal substance and the justices' backgrounds.

The system is designed with a focus on temporal validation, advanced training techniques, and robust hyperparameter optimization to ensure the model is both predictive and methodologically sound.

## 🎯 Core Objective

The central goal of this model is to predict the distribution of votes for a given Supreme Court case (e.g., percentage in favor, against, absent). 

The underlying hypothesis is that the interaction between a justice's life experience (encoded in their biography) and the specifics of a case (encoded in its description) contains powerful predictive signals about their likely vote.

## 🏗️ Model Architecture (`scotus_voting_model.py`)

The model employs a dual-encoder architecture that processes case and justice information separately before fusing them through a cross-attention mechanism. This design allows the model to learn rich, contextualized representations of how a particular court might approach a specific case.

![SCOTUS AI Model Architecture](https://storage.googleapis.com/agent-tools-prod.appspot.com/tool-results/v1/files/b0e51785-05e8-4687-84ed-da6f22822002)

**1. Dual SentenceTransformer Encoders**
Two separate, pre-trained `SentenceTransformer` models are used as encoders. They are fine-tuned during training to adapt them to the specific nuances of legal and biographical text.
-   **Biography Encoder**: Processes the textual biographies of all justices presiding over a case. It can be initialized with weights from our [Contrastive Justice Pretraining](../pretraining/README.md) for improved performance.
-   **Case Description Encoder**: Processes the textual summary of the case.

**2. NEFTune Regularization**
During training, **Noisy Embedding Fine-Tuning (NEFTune)** is applied to the outputs of the justice encoder. This technique adds a small amount of uniform noise to the embeddings, which has been shown to improve model performance and robustness by creating a more challenging optimization landscape.

**3. Justice Cross-Attention (`justice_cross_attention.py`)**
This is a key innovative component of the architecture. Instead of simply concatenating embeddings, a **multi-head cross-attention** mechanism is used.
-   The **case embedding** acts as the *query*.
-   The set of **justice embeddings** for that case act as the *keys* and *values*.
This allows the model to learn a "court representation" that is specifically tailored to the case at hand, effectively asking, "Which aspects of these justices' backgrounds are most relevant to the legal questions in this specific case?"

**4. Prediction Head**
The final, contextualized representation is passed through a Multi-Layer Perceptron (MLP) with `LayerNorm`, `ReLU` activations, and `Dropout` to produce the final output.

**5. Output Layer**
A `Softmax` activation function converts the final logits into a 4-dimensional probability distribution representing the predicted vote percentages for:
1.  Majority In Favor
2.  Majority Against
3.  Majority Absent/Recused
4.  Other (e.g., tie)

## 💡 One-Shot Learning and Hypothetical Courts

A significant innovation of this project is the model's ability to function as a **one-shot learner** for new and unseen justices. This capability stems directly from the architectural decision to use a dedicated justice encoder.

**Zero-Shot Prediction for New Justices:**
Because the model learns a generalized mapping from a justice's biography to a meaningful embedding space, it does not need to be retrained when a new justice is appointed to the Court. As long as a biography containing their pre-confirmation career is available, the model can generate an embedding for them and immediately include them in predictions for new cases. This is crucial for the model's real-world applicability, as it can adapt to changes in the Court's composition without a costly retraining cycle.

**Simulating Hypothetical Courts:**
This architectural choice unlocks powerful analytical possibilities:
-   **Hypothetical Appointments**: The model can be used to explore "what-if" scenarios. By providing the biography of any individual (e.g., a judge from a lower court, a legal scholar, or even a historical figure), one can simulate how their presence on the Court might influence the outcomes of past or future cases.
-   **Analyzing Imaginary Cases**: Similarly, the model can predict outcomes for hypothetical legal cases that have never been argued, providing insights into how a specific court might react to novel legal questions.

This transforms the model from a simple historical predictor into a flexible tool for forward-looking legal and political analysis.

## 📈 Training & Evaluation (`model_trainer.py`)

The training process is designed to be robust and incorporates several advanced techniques to ensure the model generalizes well to unseen data.

**1. Loss Function (`losses.py`)**
The primary loss function is **Kullback-Leibler (KL) Divergence Loss**. This is a deliberate choice over a more common cross-entropy loss because it measures the "distance" between two probability distributions. It is ideal for this task, as it penalizes the model not just for being wrong, but for being "confidently wrong."

**2. Fine-Tuning Strategy**
The `SentenceTransformer` encoders are initially frozen. They can be selectively unfrozen at a specified epoch (`UNFREEZE_AT_EPOCH`) and fine-tuned with a separate, smaller learning rate (`SENTENCE_TRANSFORMER_LEARNING_RATE`). This two-stage training process stabilizes learning and allows the model to first learn the task structure before adapting the deep language representations.

**3. Temporal Validation**
Crucially, the dataset is split based on time. The model is trained on older cases and validated/tested on more recent ones. This simulates a real-world prediction scenario and provides a much more rigorous evaluation than a random shuffle split, which can lead to data leakage from the future.

## 🔮 Hyperparameter Optimization (`hyperparameter_optimization.py`)

A comprehensive hyperparameter tuning pipeline using **Optuna** is a cornerstone of this project. It allows for systematic exploration of the vast hyperparameter space to find the optimal model configuration.

**1. Multi-Objective Optimization**
The optimization metric is a carefully designed composite score that balances two key objectives:
\[ \text{Metric} = \frac{\text{KL Divergence Loss} + (1 - \text{F1-Score Macro})}{2} \]
This ensures the search process finds a model that is both a good probabilistic forecaster (low KL divergence) and a strong classifier (high F1-score).

**2. Time-Based Cross-Validation**
For robust hyperparameter evaluation, the tuner employs **Time-Based Cross-Validation**. Instead of a single validation set, it creates multiple training/validation folds, each progressing forward in time. This ensures that hyperparameters are selected based on their ability to consistently generalize to future, unseen data.

**3. Selective & Parallel Tuning**
The `config.env` file provides granular control over which parameters are tuned (`TUNE_*` flags). This allows for focused experiments (e.g., "tune only the architecture" or "tune only the learning rate"). The optimization can also be parallelized across multiple jobs (`--n-jobs`).

## 🚀 How to Run

**1. Configuration (`config.py`, `config.env`)**
All parameters are controlled by `scripts/models/config.env`. This is the single source of truth for model architecture, training parameters, and optimization settings.

**2. Standard Training (`run_training.py`)**
To train a model using the parameters defined in `config.env`:
```bash
# Set a unique name for logging purposes
python scripts/models/run_training.py --experiment-name "roberta_large_run_1"
```

**3. Hyperparameter Optimization**
To launch an Optuna study:
```bash
# Run a 100-trial study with a specific name
python scripts/models/hyperparameter_optimization.py --experiment-name "full_tuning_v3" --n-trials 100
```

## 📚 References

-   **Sentence-Transformers**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
-   **Attention Mechanism**: Vaswani, A., et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
-   **NEFTune**: Jain, N., et al. (2023). *NEFTune: Noisy Embeddings Improve Instruction Finetuning*. [arXiv:2310.05914](https://arxiv.org/abs/2310.05914).
-   **Optuna**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. [arXiv:1907.10902](https://arxiv.org/abs/1907.10902).
-   **AdamW Optimizer**: Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
-   **Hugging Face Transformers**: Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. [Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations](https://www.aclweb.org/anthology/2020.emnlp-demos.6/). 