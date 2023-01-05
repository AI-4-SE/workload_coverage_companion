from cgitb import small
import pandas as pd
import networkx as nx

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import itertools
from .load import *

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

def jaccard(s1, s2):
    if type(s1) is not set:
        s1 = set(s1)
    if type(s2) is not set:
        s2 = set(s2)

    try:
        res = len(s1.intersection(s2)) / len(s1.union(s2))
    except ZeroDivisionError:
        res = 0.0
    return res

pos_color = '#C0EDA6'
neg_color = '#FF8080'

#@st.cache
def learn_model(system: str, workload: str, kpi):
    sample = load_sample(system)
    performance = load_performance(system, workload, kpi)

    X = sample.sort_values('config_id').drop(columns='config_id')
    X = remove_multicollinearity(X)
    y = performance.sort_values('config_id')[kpi]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    
    effects = (model.coef_ - np.mean(model.coef_)) / np.std(model.coef_)
    effects = pd.DataFrame([effects], columns=X.columns)
    return model, effects



def draw_dendrogram(system: str, option: str, workloads, kpi):

    workloads = sorted(workloads)

    codes = {
        workload: load_coverage(system, option, workload) for workload in workloads
    }
    sets = {w: set(list(map(str, codes[w].values))) for w in codes}

    sim_matrix = pd.DataFrame(
        np.zeros(shape=(len(sets), len(sets))), 
        columns = sets.keys(),
        index = sets.keys()
    )
    for s1, s2 in itertools.combinations(sets.keys(), 2):
        sim_matrix.loc[s1, s2] = jaccard(sets[s1], sets[s2])
        sim_matrix.loc[s2, s1] = jaccard(sets[s1], sets[s2])

    dissimilarity = 1 - abs(sim_matrix)
    for col in dissimilarity.columns:
        dissimilarity.loc[col, col] = 0
    Z = linkage(squareform(dissimilarity), 'complete')

    mlem = 6
    ratio = (len(workloads) /3) /mlem
    scale = 6
    figure, ax = plt.subplots(1, 2, figsize=(scale, ratio * scale))#, sharey=True)

    dendrogram(
        Z, 
        labels=workloads, 
        orientation='right',
        leaf_rotation=90, ax=ax[1]
    )

    ax[1].set_xlabel('Jaccard Distance')
    ax[1].set_xlim(-0.1, 1.1)

    ax[0].set_xlabel('Performance Influence')

    workloads = [www.get_text() for www in ax[1].get_yticklabels()]

    effects = []
    for workload in workloads:
        _,  e = learn_model(system, workload, kpi)
        effects.append(e)
    
    effects = pd.concat(effects)
    effect = effects[option]

    #ticks1 = ax[1].get_yticks()
    ax[1].set_yticklabels([])
    # 
    mask1 = effect < 0
    colors = []
    for i in mask1:
        if i: 
            colors.append(pos_color)
        else:
            colors.append(neg_color)

    ax[0].axvline(0, color='black', alpha=0.06)
    for workload in workloads:
        ax[0].axhline(workload, color='black', alpha=0.1, zorder=-1)
        
    # shorten workloads to 10 characters
    lebels = list(map(lambda x: x if len(x) < 11 else x[:10] + '...', workloads))
    ax[0].barh(workloads, effect, color=colors)
    
    os.system(f'mkdir -p cache/{system}/')
        
    return figure


def plot_performance_model(system, workloads, kpi, RQ1_LIMIT):
    fig = plt.figure(figsize=(2.5,4))
    plt.axvspan(-1*RQ1_LIMIT, RQ1_LIMIT, alpha=0.1, color='red')
    plt.axvline(0, color='black', alpha=0.1, linewidth=0.5)
    plt.title(f'{system}')

    coefficients = []
    for workload in workloads:
        model, coefs = learn_model(system, workload, kpi)
        coefficients.append(coefs.values)
        features = model.feature_names_in_

    coefficients = np.vstack(coefficients)
    coefficients = pd.DataFrame(coefficients, columns=features, index=workloads)
    means = np.mean(coefficients, axis=0)
    mean_dict = {features[i]: means[i] for i, f in enumerate(features)}
    mean_dict = {k: v for k, v in sorted(mean_dict.items(), key=lambda item: item[1])}

    maxx = coefficients.abs().max() < RQ1_LIMIT
    maxx = np.argwhere(maxx.values == True)
    features = features[maxx].ravel()
    for feature in features:
        del mean_dict[feature]

    for col in mean_dict.keys():
        plt.axhline(col, alpha=0.1, color='black', linewidth=0.5, zorder=-1)

    for workload in workloads:
        plt.scatter(coefficients.loc[workload, mean_dict.keys()], mean_dict.keys(), marker='|', color='black', s=70)

    # coloring for bars
    xmin, xmax = plt.xlim()
    xrange = xmax - xmin
    zero = (0 - xmin) / xrange
    for option in coefficients.columns:
        smallest_negative = coefficients[coefficients < 0][option].min()
        largest_negative = coefficients[coefficients < 0][option].max()

        smallest_negative = (smallest_negative - xmin) / xrange
        largest_negative = (largest_negative - xmin) / xrange

        smallest_positive = coefficients[coefficients > 0][option].min()
        largest_positive = coefficients[coefficients > 0][option].max()

        smallest_positive = (smallest_positive - xmin) / xrange
        largest_positive = (largest_positive - xmin) / xrange

        df = coefficients[option]
        if df[df < 0].shape[0] == 1 or df[df > 0].shape[0] > 0:
            plt.axhline(option, smallest_negative, zero, color=neg_color, linewidth=8, alpha=1, zorder=0, solid_capstyle="butt")
        else:
            plt.axhline(option, smallest_negative, largest_negative, color=neg_color, linewidth=8, alpha=1, zorder=0, solid_capstyle="butt")

        if df[df > 0].shape[0] == 1 or df[df < 0].shape[0] > 0:
            plt.axhline(option, zero, largest_positive, color=pos_color, linewidth=8, alpha=1, zorder=0, solid_capstyle="butt")
        else:
            plt.axhline(option, smallest_positive, largest_positive, color=pos_color, linewidth=8, alpha=1, zorder=0, solid_capstyle="butt")
    
    plt.xlabel('Standardized Performance Influence')

    return fig


def cache_model_plots(system, workloads, kpi):
    
    coefficients = []
    for workload in workloads:
        model, coefs = learn_model(system, workload, kpi)
        coefficients.append(coefs.values)
        features = model.feature_names_in_

    coefficients = np.vstack(coefficients)
    coefficients = pd.DataFrame(coefficients, columns=features, index=workloads)
    
    for col in coefficients.columns:
        os.system(f'mkdir -p cache/{system}/')
        fig = plt.figure(figsize=(4,1))
        
        plt.axhline(0, alpha=0.1, color='black')
        plt.axvline(0, alpha=0.1, color='black')
        plt.scatter(coefficients.loc[:,col], np.zeros(coefficients.shape[0]), marker='|', color='black', s=50)
        plt.savefig(f'cache/{system}/{col}.pdf', bbox_inches='tight' )
    
    pass

def correlation_analysis(system, workloads, kpi, PEARSON_LIMIT, KENDALL_LIMIT):
    pearson_correlation = pd.DataFrame(
        data=np.ones(shape=(len(workloads), len(workloads))),
        index=workloads,
        columns=workloads
    )

    kendall_correlation = pd.DataFrame(
        data=np.ones(shape=(len(workloads), len(workloads))),
        index=workloads,
        columns=workloads
    )

    for w1, w2 in itertools.combinations(workloads, 2):
        p1 = load_performance(system, w1, kpi).sort_values('config_id')[kpi].values
        p2 = load_performance(system, w2, kpi).sort_values('config_id')[kpi].values

        pearsonr = stats.pearsonr(p1, p2)[0]
        kendallt = stats.kendalltau(p1, p2)[0]

        pearson_correlation.loc[w1, w2] = pearsonr
        pearson_correlation.loc[w2, w1] = pearsonr

        kendall_correlation.loc[w1, w2] = kendallt
        kendall_correlation.loc[w2, w1] = kendallt

    pearson_correlation = np.abs(pearson_correlation)
    kendall_correlation = np.abs(kendall_correlation)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10,4.5))
    plt.suptitle(f'Correlation Analaysis for "{system}"')
    cmap = 'coolwarm'

    mask_pearson = pearson_correlation < PEARSON_LIMIT
    mask_kendall = kendall_correlation < KENDALL_LIMIT

    sns.heatmap(data=pearson_correlation, vmin=0, vmax=1, cmap=cmap, ax=ax[0], mask=mask_pearson)
    sns.heatmap(data=kendall_correlation, vmin=0, vmax=1,  cmap=cmap, ax=ax[1], mask=mask_kendall)

    ax[0].set_title(f"Pearson's r (r > {PEARSON_LIMIT})")
    ax[1].set_title(f"Kendall's t (t > {KENDALL_LIMIT})")
    
    lt = 0
    xmt = 0
    nmt = 0

    for w1, w2 in itertools.combinations(workloads, 2):
        if pearson_correlation.loc[w1, w2] >= PEARSON_LIMIT:
            lt += 1
        elif pearson_correlation.loc[w1, w2] < PEARSON_LIMIT and kendall_correlation.loc[w1, w2] >= KENDALL_LIMIT:
            xmt += 1
        else:
            nmt += 1

    sum_classes = sum([lt, nmt, xmt])
    classification = pd.DataFrame(data=[[lt, round(lt/sum_classes * 100, 1),  xmt, round(xmt/sum_classes * 100, 2), nmt, round(nmt/sum_classes * 100, 2)]], columns = ['LT','LT [%]', 'XMT', 'XMT [%]', 'NMT', 'NMT [%]'])


    return fig, classification


def distribution_plot(system, workloads, kpi, standardized: bool):
    fig, ax = plt.subplots(2, 1, figsize=(9,8))
    for workload in workloads:
        df = load_performance(system, workload, kpi)
        df = df[kpi].values
        if standardized:
            df = (df - np.mean(df)) / np.std(df)
        sns.kdeplot(df, bw=0.03, label=workload, ax=ax[0])
        ax[1].plot(sorted(df), label=workload)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel(f'{kpi}')
    return fig



def remove_multicollinearity(df: pd.DataFrame):

    # remove columns with identical values (dead features or mandatory features)
    nunique = df.nunique()
    mandatory_or_dead = nunique[nunique == 1].index.values

    df = df.drop(columns=mandatory_or_dead)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df_configs = df
    alternative_ft = []
    alternative_ft_names = []
    group_candidates = {}

    for i, col in enumerate(df_configs.columns):
        filter_on = df_configs[col] == 1
        filter_off = df_configs[col] == 0
        group_candidates[col] = []
        for other_col in df_configs.columns:
            if other_col != col:
                values_if_col_on = df_configs[filter_on][other_col].unique()
                if len(values_if_col_on) == 1 and values_if_col_on[0] == 0:
                    # other feature is always off if col feature is on
                    group_candidates[col].append(other_col)

    G = nx.Graph()
    for ft, alternative_candidates in group_candidates.items():
        for candidate in alternative_candidates:
            if ft in group_candidates[candidate]:
                G.add_edge(ft, candidate)

    cliques_remaining = True
    while cliques_remaining:
        cliques_remaining = False
        cliques = nx.find_cliques(G)
        for clique in cliques:
            # check if exactly one col is 1 in each row
            sums_per_row = df_configs[clique].sum(axis=1).unique()
            if len(sums_per_row) == 1 and sums_per_row[0] == 1.0:
                delete_ft = sorted(clique)[0]
                alternative_ft_names.append(delete_ft)
                df_configs.drop(delete_ft, inplace=True, axis=1)
                for c in clique:
                    G.remove_node(c)
                cliques_remaining = True
                break

    return df
