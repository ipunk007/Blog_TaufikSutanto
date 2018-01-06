# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 02:59:44 2017
Depth First Search, asumsi urutan = list di V
http://blog-sutanto.blogspot.co.id/2018/01/bfs-dfs-dengan-networkx-di-python.html
@author: Taufik Sutanto
"""
import networkx as nx
from my_NetworkX_lib import buildGraph, drawGraph

V = ['a', 'b', 'c', 'd' , 'e', 'f', 'g', 'h']
E = [('a','b'),('a','c'),('a','g'),('a','g'),('b','d'),('b','g'),('c','d'),('c','e'),('d','f'),('e','f'),('e','f'),('f','h'),('g','e')]
G = buildGraph(V,E=E)
root = V[0]

VDFS = V[:] # a copy of V
DFS = nx.Graph() # empty graph
v = VDFS[0]; del VDFS[0]
stack = [root] # untuk backtrack
while VDFS: # VDFS not empty
    DFS.add_node(v)
    e = G.edges(v)
    nextNode = [v2 for v1,v2 in e if v2 not in DFS.nodes()]
    if nextNode: # Not Empty
        nextNode.sort() # meyakinkan urutannya sesuai urutan node (abjad)
        DFS.add_node(nextNode[0])
        DFS.add_edge(v,nextNode[0])
        v = nextNode[0]
        stack.append(v)
        del VDFS[VDFS.index(v)]
    else: #Backtrack
        v = stack[-1]; del stack[-1]
        
drawGraph(G, file="GraphAwal.png", labels = True, edge_labels=False, gType = 'spring')
drawGraph(DFS, file="DFS.png", labels = True, edge_labels=False, gType = 'spring')
