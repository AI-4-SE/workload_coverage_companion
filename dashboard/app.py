from multiprocessing.sharedctypes import Value
import streamlit as st

import os

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import scipy.stats as stats
import itertools

from lib.load import *
from lib.analyses import *

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


st.set_page_config(
        page_title="Dashboard for ICSE submission #177",
        page_icon="🔎",
        layout="wide",
    )

# globals
KPI = {
    'z3': 'time',
    'x264': 'time',
    'lrzip': 'time',
    'xz': 'time',
    'dconvert': 'time',
    'jump3r': 'time',
    'batik': 'time',
    'kanzi': 'time',
    'h2': 'throughput'
}

TABS = st.tabs([
    "Getting Started",
    "Performance Distributions (RQ1)", 
    "Performance Influences (RQ2)",
    "Code Coverage Analysis (RQ3)"
])



# show controls for all visualizations
with st.sidebar:

    SYSTEM = st.selectbox(
        label = 'Software System', 
        options = sorted(KPI.keys())
    )

    st.markdown("-------")
    st.markdown('#### Settings for Analyses')

    WORKLOADS = st.multiselect(
    label = 'Workload', 
    options = load_workloads(SYSTEM)
    )
        
    all_workloads = st.checkbox("Select all workloads?", True)

    if all_workloads:
        WORKLOADS = load_workloads(SYSTEM)

    with st.expander("RQ1 (Performance Distributions)"):
        PEARSON_LIMIT = st.slider("Threshold (Pearson's r)", 0.0, 1.0, 0.6, 0.01,)
        KENDALL_LIMIT = st.slider("Threshold (Kendall's t)", 0.0, 1.0, 0.6, 0.01,)

    with st.expander("RQ3 (Code Coverage Analysis)"):
        OPTION = st.selectbox(
            label = 'Software Option / Feature', 
            options = load_options(SYSTEM, KPI[SYSTEM])
        )


# Landing Page    #lebels = [item.get_text() for item in ax[1].get_xticklabels()]
    #plt.yticks(plt.yticks(), lebels)
    #st.code(lebels)

with TABS[0]:
    st.markdown('''
    
    ## Interactive Dashbaord for Data Exploration
    
    This dashbaord is part of the supplementary material to the ICSE 2023 paper *"Analyzing the Impact of Workloads on Modeling
    the Performance of Configurable Software Systems"*. While we provide our experiment data via the companion [Gitlab repository]() and [Cloud Folder](), this dashbaord is intended  
    to provide a means to reenact our analysis & findings as well as explore visualizations not included in the paper submission due to space constraints. 

    Below, we provide a small walkthrough on how to use the dashboard. Specifically, what parameters one can select for each analysis. In total, this dashboard provides four tabs (you're seeing the first one, 
    the remaining ones refer to one research question each.     ''')

    st.markdown('''
    ### Sidebar
    On the left, you can find a couple of options to selct from. The first selection is the **Software System** which you want to explore data for. This selection refers to the nine subject systems studied. 

    As some analyses study performance/properties across workloads, one can select the workloads of interest in the multi-selectbox **Workload(s)**.
    ''')

# show visualizations for correlation analysis (RQ1)
with TABS[1]:
    columns = st.columns([1,0.05, 1])
    with columns[0]:
        fig, classification = correlation_analysis(SYSTEM, WORKLOADS, KPI[SYSTEM], PEARSON_LIMIT, KENDALL_LIMIT)
        st.pyplot(fig)
        st.write(classification)

    with columns[2]:
        st.markdown(f'''
        #### Explanation
        The two heatmaps provide an overview over the relationships between the configuration-dependent 
        performance distributions across different workloads of the software system **{SYSTEM}**. The left heatmap refers to Pearson's *r* correlation
        coefficient and the right heatmap refers to Kendall's *t* rank correlation coefficient. In both heatmaps, relationships below an 
        absolute value of **{PEARSON_LIMIT}** (Pearson) or **{KENDALL_LIMIT}** (Kendall), respectively, are masked. These metrics are used to classify
        pairs of workloads into three categories (LT: linear relationship, XMT: monotonous relationship, and NMT: non-monotonous 
        relationship). We provide the categorization counts in a table below (cf. Table III in the paper). 

        #### Settings
        You can adjust the absolute threshold values for Pearson's *r* and Kendall's *t* in the two sliders by expanding 
        the panel *'RQ1 (Performance Distributions)'*.
        
        ''')

# show visualizations for performance models (RQ2)
with TABS[2]:
    columns = st.columns([1,0.05, 1])
    with columns[0]:

        fig = plot_performance_model(SYSTEM, WORKLOADS, KPI[SYSTEM], 0)
        st.pyplot(fig)

    with columns[2]:
        st.markdown(f'''
        #### Explanation
        This view provides an overview over the learned performance influences for each 
        configuration option of the selected software system **{SYSTEM}**. These graphics are the same as the ones provided in Figure 2
        in the paper, where we limit the presentation to jump3r, h2, and z3.

        Each vertical bar depicts the performance influence under a specific workload. The colored ranges depict positive (green) and
        negative (red) performance influences. To study, how each configuration option influences performance under a specific workload, switch to the next tab.
        ''')
    
# show visualizations for code coverage analysis (RQ3)
with TABS[3]:
    columns = st.columns([1,0.05, 1])
    with columns[0]:
        try:
            figure = draw_dendrogram(SYSTEM, OPTION, WORKLOADS, KPI[SYSTEM])
            st.pyplot(figure)
        except:
            st.markdown(''' 
            (No graphic available)
            ''')

    with columns[2]:
        st.markdown(f'''
        #### Explanation

        ##### Workload-dependent Performance Influence
        This view on the **left** illustrates how the performance influence of configuration option **{OPTION}** varies across workloads. You can select
        the configuration option of infterest under the select box in the sidebar under *RQ3 (Code Coverage Analysis)*. The graphics generated
        here refer to Figure 3 in the paper and are provided for most configuration options.

        There are two reasons, a graphic is not available for everey configuration option:
        
        * Some configuration options might be dropped during preprocessing when learning an explanatory model to avoid multicollinearity.
        * For numeric configuration options, we cannot provide option code as numeric features cannot be enabled or disabled, only varied.

        ##### Workload-dependent Option-Code Coverage
        The **right** side of this graphic (dendrogram) illustrates illustrate how similar the covered
        lines of option-specific code under each workload are. The dendrograms depict the Jaccard similarity clustering, where
        (from left to right) the split points indicate what Jaccard distance individual sets of lines or subclusters exhibit.
         The further to the left the point is, the more similar are the constituent parts.
        ''')

  
