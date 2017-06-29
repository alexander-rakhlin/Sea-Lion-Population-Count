import pandas as pd
import pickle
import numpy as np

SUBMISSION_FILE = "submission/submission.0.50.csv"
PRED_DICT_FILES = (
    "submission/preds_dict_test.0.80-inf.pkl",
    "submission/preds_dict_test.0.75-0.80.pkl",
    "submission/preds_dict_test.0.70-0.75.pkl",
    "submission/preds_dict_test.0.65-0.70.pkl",
    "submission/preds_dict_test.0.60-0.65.pkl",
    "submission/preds_dict_test.0.55-0.60.pkl",
    "submission/preds_dict_test.0.50-0.55.pkl",
)

HEADER = ["test_id", "adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

df = None
for PRED_DICT_FILE in PRED_DICT_FILES:
    with open(PRED_DICT_FILE, mode="rb") as f_:
        m_pred, m_direct = pickle.load(f_)
    df_ = pd.DataFrame.from_dict(m_pred, orient="index")
    df_.columns = HEADER[1:]
    df_.index.name = HEADER[0]
    df = df_ if df is None else df + df_

df = df.apply(lambda x: np.maximum(0, np.round(x))).astype(int)
df[HEADER[1:]].to_csv(SUBMISSION_FILE, index=True, header=True)
