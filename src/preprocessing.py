import pandas as pd
from sklearn.

#assessing and cleaning data, also loading:)
def load_and_clean_data(filepath):
    df = pd.read_cvs(filepath).dropna()

#encoding certain variable types
labeling = {}
