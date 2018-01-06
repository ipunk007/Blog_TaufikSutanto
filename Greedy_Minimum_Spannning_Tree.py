# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:20:42 2018
Teori Graph di Python lewat NetworkX: Studi Kasus Greedy Algorithm
http://taufiksutanto.blogspot.co.id/2018/01/teori-graph-di-python-lewat-networkx.html
@author: Taufik Sutanto
"""
import matplotlib.pyplot as plt
import networkx as nx
V = ['a','b','c','d','e','f','g','z'] # Vertices
E = [('a','b',2),('a','f',1),('b','c',2),('b','d',2),
     ('b','e',4),('c','e',3),('c','z',1),('d','e',4),('d','f',3),
     ('e','g',7),('f','g',5),('g','z',6)] # Edges & Weight-nya
# Mulai Membentuk Graph-nya
G = nx.Graph() # Empty Graph
for vertex in V: # Menambahkan semua vertexnya
    G.add_node(vertex)
for v1,v2,w in E: # Menambahkan edgesnya
    G.add_edge(v1,v2, weight=w)
print('G: {0} nodes, {1} edges'.format(G.number_of_nodes(),G.number_of_edges()))    

# start drawing graph G
plt.figure(1) # Biar graphnya ndak numpuk dengan yang setelahnya
pos = nx.spring_layout(G)
eL = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_nodes(G,pos) 
nx.draw_networkx_labels(G,pos)
nx.draw_networkx_edge_labels(G,pos,edge_labels=eL)
nx.draw_networkx_edges(G,pos)

# Solusi Greedy
MST = nx.Graph() # Empty Graph as G
for vertex in V: # Menambahkan semua vertex di G ke MST
    MST.add_node(vertex)
while E: # sama saja dengan While len(W)>0: 
    # sengaja whilenya seperti ini biar nampak Pythonista banget :) 
    Wmin = min(E, key = lambda t: t[2])[-1] # Ga efisien nih disini, tapi gapapa ya.. kan tutorial dasar
    idx = [w[2] for w in E].index(Wmin) # ini juga sebenarnya bukan yang paling efisien
    Emin = E[idx][:2] # hanya ambil informasi edgenya
    del E[idx] # hapus edge minimal tersebut dari E
    MST.add_edge(Emin[0],Emin[1], weight=Wmin) # Tambahkan edge minimal ke MST
    if len(nx.cycle_basis(MST,'a'))>0: # Check timbul cycle atau tidak, misal 'a' root dari tree-nya
        MST.remove_edge(Emin[0],Emin[1]) # Undo jika iya

# start drawing graph MST
plt.figure(2) # Biar graphnya ndak numpuk dengan yang sebelumnya
pos = nx.spring_layout(MST)
eL = nx.get_edge_attributes(MST,'weight')
nx.draw_networkx_nodes(MST,pos) 
nx.draw_networkx_labels(MST,pos)
nx.draw_networkx_edge_labels(MST,pos,edge_labels=eL)
nx.draw_networkx_edges(MST,pos)
