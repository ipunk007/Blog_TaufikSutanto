# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 02:59:44 2017
Breadth First Search, asumsi urutan = list di V
http://blog-sutanto.blogspot.co.id/2018/01/bfs-dfs-dengan-networkx-di-python.html
@author: Taufik Sutanto
"""
import networkx as nx
from my_NetworkX_lib import buildGraph, drawGraph

V = ['a', 'b', 'c', 'd' , 'e', 'f', 'g', 'h']
E = [('a','b'),('a','c'),('a','g'),('a','g'),('b','d'),('b','g'),('c','d'),('c','e'),('d','f'),('e','f'),('e','f'),('f','h'),('g','e')]
G = buildGraph(V,E=E)
root = V[0]

Levels = [[root]]
Vtemp = [root]
for v in V:
    e = G.edges(v)
    e = [v2 for v1,v2 in e if v2 not in Vtemp]
    if e:
        Levels.append(e)
        Vtemp = Vtemp + e
BFS = buildGraph(V)
for vertices in Levels:
    for v in vertices:
        e = G.edges(v)
        for v1, v2 in e:
            BFS.add_edge(v1,v2)
            if len(nx.cycle_basis(BFS,root))>0:
                BFS.remove_edge(v1,v2)
                
drawGraph(G, file="GraphAwal.png", labels = True, edge_Label=False, gType = 'spring')
drawGraph(BFS, file="BFS.png", labels = True, edge_Label=False, gType = 'spring')
