# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:55:48 2017
My Personal NetworkX Functions
@author: Taufik Sutanto
"""
import matplotlib.pyplot as plt, networkx as nx

def buildGraph(V,E=None):
    G = nx.Graph()
    for v in V:
        G.add_node(v)
    if E:
        if len(E[0])>2:
            for v1, v2, w in E:
                G.add_edge(v1,v2, weight=w)
        else:
            for v1, v2 in E:
                G.add_edge(v1,v2)
    return G

def drawGraph(G, file ='graph.png',labels=False,edge_Label=False, gType = 'spring'):
    fig1 = plt.figure(); fig1.add_subplot(111)
    if gType == 'spring':
        pos = nx.spring_layout(G)
    elif gType == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spectral_layout(G)
    nx.draw_networkx_nodes(G,pos)
    if labels:
        nx.draw_networkx_labels(G,pos)
    if edge_Label:
        nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'weight'))
    nx.draw_networkx_edges(G,pos)
    plt.savefig(file);  plt.show()
