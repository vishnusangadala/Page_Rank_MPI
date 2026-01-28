from mpi4py import MPI
import os
from collections import defaultdict

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Paths
BASE_DIR = "/gpfs/projects/AMS598/projects2025_submission/project2_submission/Sangadala_Vishnu/"
DATA_FILE = "/gpfs/projects/AMS598/projects2025_data/project2_data/graph.txt"
OUT_DIR = BASE_DIR + "output/"
INT_DIR = BASE_DIR + "intermediate/"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INT_DIR, exist_ok=True)

beta = 0.9
iterations = 4

# Read data (root)
edges = []
nodes = set()

if rank == 0:
    with open(DATA_FILE, "r") as f:
        for line in f:
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

edges = comm.bcast(edges, root=0)
nodes = comm.bcast(nodes, root=0)

N = len(nodes)
pagerank = {n: 1.0 / N for n in nodes}

# Build adjacency list
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)

# PageRank iterations
for it in range(iterations):
    local_pr = defaultdict(float)

    for i, u in enumerate(adj):
        if i % size == rank:
            out_deg = len(adj[u])
            if out_deg > 0:
                share = pagerank[u] / out_deg
                for v in adj[u]:
                    local_pr[v] += share

    all_pr = comm.gather(local_pr, root=0)

    if rank == 0:
        new_pr = {n: (1 - beta) / N for n in nodes}
        for part in all_pr:
            for k, v in part.items():
                new_pr[k] += beta * v
        pagerank = new_pr

    pagerank = comm.bcast(pagerank, root=0)

# Output top 10
if rank == 0:
    top10 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    with open(OUT_DIR + "top10_pagerank.txt", "w") as f:
        for node, score in top10:
            f.write(f"{node}\t{score}\n")
