import os

# PARAMETERS
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_BASE = "microsoft/deberta-v3-base"
# MODEL_BASE = "microsoft/deberta-v3-large"
MAX_LENGTH_TOKENIZER = 256
N_FOLDS = 5
EPOCHS = 3
BATCH_SIZE = 32             # Try to use multiples of 8 for maximum GPU efficiency
LEARNING_RATE = 1e-5
EVAL_STEPS = 250           # Try to use an integer divisor of the total number of steps: total_number_steps = round_up(dataset_n_rows / BATCH_SIZE) * EPOCHS
SAVE_STEPS = 250
SAVE_TOTAL_LIMIT = 1
EARLY_STOP_PATIENCE = 4
LOGGING_STEPS = 10
RANDOM_SEED = 31415

# PATHS
PATH_BASE_DIR = os.path.dirname(__file__)
PATH_DATA = os.path.join(PATH_BASE_DIR, "data")
PATH_CHECKPOINTS = os.path.join(PATH_BASE_DIR, "model_checkpoints")
PATH_RESULTS = os.path.join(PATH_BASE_DIR, "results")

PATH_DF_TRAIN = os.path.join(PATH_DATA, "original", "train.csv")
PATH_DF_TEST = os.path.join(PATH_DATA, "original", "test.csv")
PATH_DF_TEST_LABELS = os.path.join(PATH_DATA, "original", "test_labels.csv")

PATH_DATA_PROCESSED_DIR = os.path.join(PATH_DATA, "processed", MODEL_BASE)
PATH_DS_TRAIN_TOKENIZED = os.path.join(PATH_DATA_PROCESSED_DIR, "ds_train_tokenized")
PATH_DS_TEST_TOKENIZED = os.path.join(PATH_DATA_PROCESSED_DIR, "ds_test_tokenized")