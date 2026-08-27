from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "train_series.csv"

df = pd.read_csv(DATA_PATH)

st.title("MRI Pipeline")
st.dataframe(df)

