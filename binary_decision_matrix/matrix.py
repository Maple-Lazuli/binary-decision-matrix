import matplotlib.pyplot as plt
import networkx as nx
import itertools
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox


def compare(opt1, opt2, criteria_name=None):
    if criteria_name is not None:
        question = f'{opt1[1]} more favorable than {opt2[1]} regarding {criteria_name}?'
        response = messagebox.askquestion(criteria_name, question)
        swap = False if response == 'yes' else True
    else:
        question = f'{opt1[1]} is more important than {opt2[1]}'
        response = messagebox.askquestion("Criteria Ordering", question)
        swap = False if response == 'yes' else True
    if swap:
        temp = opt2[1]
        opt2[1] = opt1[1]
        opt1[1] = temp
    opt1[2].append(opt2[1])
    opt2[2].append(opt1[1])
    return opt1, opt2, swap


def check(opt1, opt2, criteria_name=None, use_history=True):
    if use_history:
        if (opt1[1] in opt2[2]) and (opt2[1] in opt1[2]):
            return opt1, opt2, False
    return compare(opt1, opt2, criteria_name)


def get_plot(options, decisions, criteria_name, ax):
    plt.ioff()
    red_edges = []
    for idx in range(len(decisions) - 1, 0, -1):
        red_edges.append((decisions[idx][1], decisions[idx - 1][1]))

    all_edges = []
    for subset in itertools.combinations(options, 2):
        all_edges.append(subset)

    ignore_edges = [(r[1], r[0]) for r in red_edges]

    G = nx.DiGraph()
    G.add_edges_from(all_edges)

    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in G.edges()]

    black_edges = [edge for edge in G.edges() if edge not in red_edges + ignore_edges]

    pos = nx.circular_layout(G)

    ax.set_title(f"Decisions for {criteria_name}")
    nx.draw_networkx_nodes(G, pos, node_size=200, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False, ax=ax)


def evaluate_criteria(options, criteria_name=None):
    changed = True
    changes = []
    while changed:
        for idx in range(len(options) - 1):
            opt1 = options[idx]
            opt2 = options[idx + 1]
            opt1, opt2, c = check(opt1, opt2, criteria_name)
            changes.append(c)
            options[idx] = opt1
            options[idx + 1] = opt2
        changed = sum(changes) != 0
        changes = []
    return options


def get_criteria_weight(criteria_evaled, criteria):
    criteria_values = np.linspace(1, .5, len(criteria_evaled))
    for c in criteria_evaled:
        if c[1] == criteria:
            return criteria_values[c[0]]


def get_dataframe(options, matrix_dict, criteria, criteria_evaled=None):
    values = np.linspace(1, 0, len(options))

    decision_dict = dict()
    for opt in options:
        scores = []
        for c in matrix_dict.keys():
            for c_o in matrix_dict[c]:
                if opt == c_o[1]:
                    if criteria_evaled is not None:
                        scores.append(values[c_o[0]] * get_criteria_weight(criteria_evaled, c))
                    else:
                        scores.append(values[c_o[0]])
        decision_dict[opt] = scores

    column_names = dict()
    for idx, c in enumerate(criteria):
        column_names[idx] = c
    df = pd.DataFrame.from_dict(decision_dict, orient='index')
    df.rename(columns=column_names, inplace=True)
    df['Total'] = df.sum(axis=1)
    df = df.sort_values(['Total'], ascending=False)
    return df


class DecisionMatrix:
    def __init__(self, criteria, options):
        self.options = options
        self.criteria = criteria

        self.matrix_dict = dict()
        for c in self.criteria:
            self.matrix_dict[c] = [[idx, val, []] for idx, val in enumerate(options)]

        self.criteria_evaled = None

        self.root = tk.Tk()
        self.root.withdraw()

    def decide(self):
        for c in self.matrix_dict.keys():
            self.matrix_dict[c] = evaluate_criteria(self.matrix_dict[c], c)

    def get_figure(self):
        figure_dimensions = 5
        fig, ax = plt.subplots(nrows=len(self.matrix_dict.keys()), ncols=1,
                               figsize=(figure_dimensions, figure_dimensions * len(self.matrix_dict.keys())))
        for idx, k in enumerate(self.matrix_dict.keys()):
            get_plot(self.options, self.matrix_dict[k], k, ax[idx])

        return fig

    def weigh_criteria(self):
        self.criteria_evaled = evaluate_criteria([[idx, val, []] for idx, val in enumerate(self.criteria)], None)

    def get_dataframe(self):
        return get_dataframe(self.options, self.matrix_dict, self.criteria, criteria_evaled=self.criteria_evaled)
