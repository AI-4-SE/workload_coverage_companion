import pandas as pd
import streamlit as st

@st.cache
def load_workloads(system: str):
    df = pd.read_csv(f'resources/{system}/measurements.csv')
    workloads = df.workload.unique()
    return workloads
    
@st.cache
def load_performance(system: str, workload: str, kpi: str):
    df = pd.read_csv(f'resources/{system}/measurements.csv')
    df = df[df['workload'] == workload]
    df = df.loc[:, ['config_id', kpi]]
    return df

@st.cache
def load_sample(system: str):
    df = pd.read_csv(f'resources/{system}/sample.csv')
    return df

@st.cache
def load_options(system: str, kpi: str):
    df = pd.read_csv(f'resources/{system}/sample.csv')
    options = set(df.columns) - set(['config_id', kpi])
    return sorted(options)

@st.cache
def load_coverage(system: str, option: str):
    df = pd.read_csv(f'resources/{system}/code/option_code/{option}.csv')
    return df

@st.cache
def load_coverage(system: str, option: str, workload: str):
    df = pd.read_csv(f'resources/{system}/code/workload_specific/{option}/{workload}.csv')
    return df
