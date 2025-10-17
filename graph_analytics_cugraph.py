"""graph_analytics_cugraph.py

GPU-Accelerated Graph Analytics with RAPIDS cuGraph
====================================================

Interactive graph analysis using RAPIDS cuGraph for GPU-accelerated network
analytics. Compare CPU-based NetworkX with GPU-accelerated cuGraph for
algorithms like PageRank, community detection, and shortest paths on large graphs.

Features:
- Generate synthetic graphs (scale-free, random, small-world)
- CPU (NetworkX) vs GPU (cuGraph) performance comparison
- Multiple graph algorithms (PageRank, Louvain, BFS, shortest path)
- Interactive visualization of algorithm results
- Scalability testing on graphs up to millions of edges

Requirements:
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.4+
- RAPIDS cuGraph (optional, will fall back to NetworkX)
- For million-edge graphs: 16GB+ GPU memory recommended

Author: Brev.dev Team
Date: 2025-10-17
"""

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def __():
    """Import dependencies"""
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from typing import Dict, List, Optional, Tuple
    import subprocess
    import time
    import torch
    
    # NetworkX (CPU baseline)
    try:
        import networkx as nx
        NX_AVAILABLE = True
    except ImportError:
        NX_AVAILABLE = False
        nx = None
    
    # cuGraph (GPU accelerated)
    try:
        import cugraph
        CUGRAPH_AVAILABLE = True
    except ImportError:
        CUGRAPH_AVAILABLE = False
        cugraph = None
    
    return (
        mo, np, pd, go, px, Dict, List, Optional, Tuple,
        subprocess, time, torch, nx, NX_AVAILABLE, 
        cugraph, CUGRAPH_AVAILABLE
    )


@app.cell
def __(mo, NX_AVAILABLE, CUGRAPH_AVAILABLE):
    """Title and library availability"""
    mo.md(
        f"""
        # 🕸️ GPU-Accelerated Graph Analytics
        
        **Analyze large-scale networks at GPU speed** with RAPIDS cuGraph.
        cuGraph provides NetworkX-compatible APIs that run orders of magnitude
        faster on NVIDIA GPUs.
        
        **Library Status**:
        - NetworkX (CPU): {'✅ Available' if NX_AVAILABLE else '⚠️ Not installed'}
        - cuGraph (GPU): {'✅ Available' if CUGRAPH_AVAILABLE else '⚠️ Not installed'}
        
        **Speedup Examples** (10M edge graph):
        - PageRank: 100x faster on GPU
        - Louvain (community detection): 50x faster
        - Single-Source Shortest Path: 80x faster
        - Connected Components: 150x faster
        
        ## ⚙️ Graph Configuration
        """
    )
    return


@app.cell
def __(mo):
    """Interactive graph controls"""
    graph_type = mo.ui.dropdown(
        options=['scale_free', 'random', 'small_world', 'complete'],
        value='scale_free',
        label="Graph Type"
    )
    
    num_nodes = mo.ui.slider(
        start=3, stop=6, step=1, value=4,
        label="Number of Nodes (10^x)", show_value=True
    )
    
    edge_density = mo.ui.slider(
        start=2, stop=20, step=1, value=5,
        label="Avg Edges per Node", show_value=True
    )
    
    algorithms = mo.ui.multiselect(
        options=['pagerank', 'louvain', 'connected_components', 'betweenness'],
        value=['pagerank', 'louvain'],
        label="Algorithms to Benchmark"
    )
    
    run_benchmark_btn = mo.ui.run_button(label="🚀 Run Graph Benchmark")
    
    mo.vstack([
        mo.hstack([graph_type, algorithms], justify="start"),
        mo.hstack([num_nodes, edge_density], justify="start"),
        mo.md(f"**Approximate graph size**: {10**num_nodes.value:,} nodes, {10**num_nodes.value * edge_density.value:,} edges"),
        run_benchmark_btn
    ])
    return graph_type, num_nodes, edge_density, algorithms, run_benchmark_btn


@app.cell
def __(torch, mo, subprocess):
    """GPU Detection"""
    
    def get_gpu_info() -> Dict:
        """Query NVIDIA GPU information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=5
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    idx, name, mem, compute = line.split(', ')
                    gpus.append({
                        'GPU': int(idx),
                        'Model': name,
                        'Memory (GB)': f"{int(mem) / 1024:.1f}",
                        'Compute Cap': compute
                    })
            return {'available': True, 'gpus': gpus}
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    gpu_info = get_gpu_info()
    
    if gpu_info['available']:
        mo.callout(
            mo.vstack([
                mo.md("**✅ GPU Ready for cuGraph**"),
                mo.ui.table(gpu_info['gpus'])
            ]),
            kind="success"
        )
    else:
        mo.callout(
            mo.md(f"⚠️ **No GPU detected**: {gpu_info.get('error', 'Unknown')}"),
            kind="warn"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return get_gpu_info, gpu_info, device


@app.cell
def __(mo, device, torch):
    """GPU Memory Monitor"""
    
    def get_gpu_memory() -> Optional[Dict]:
        """Get current GPU memory usage"""
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                'Allocated': f"{allocated:.2f} GB",
                'Reserved': f"{reserved:.2f} GB",
                'Free': f"{total - reserved:.2f} GB"
            }
        return None
    
    gpu_memory = mo.ui.refresh(
        lambda: get_gpu_memory(),
        options=[2, 5, 10],
        default_interval=5
    )
    
    mo.vstack([
        mo.md("### 📊 GPU Memory"),
        gpu_memory if gpu_memory.value else mo.md("*CPU mode*")
    ])
    return get_gpu_memory, gpu_memory


@app.cell
def __(nx, np, pd, NX_AVAILABLE):
    """Graph generation utilities"""
    
    def generate_graph_nx(graph_type: str, num_nodes: int, avg_degree: int):
        """Generate graph using NetworkX"""
        if not NX_AVAILABLE:
            raise ImportError("NetworkX not available")
        
        if graph_type == 'scale_free':
            # Barabási–Albert preferential attachment
            m = avg_degree // 2
            G = nx.barabasi_albert_graph(num_nodes, m)
        elif graph_type == 'random':
            # Erdős–Rényi random graph
            p = avg_degree / num_nodes
            G = nx.erdos_renyi_graph(num_nodes, p)
        elif graph_type == 'small_world':
            # Watts–Strogatz small-world
            k = avg_degree
            p = 0.1
            G = nx.watts_strogatz_graph(num_nodes, k, p)
        elif graph_type == 'complete':
            G = nx.complete_graph(num_nodes)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        return G
    
    def nx_to_edgelist(G) -> pd.DataFrame:
        """Convert NetworkX graph to edge list DataFrame"""
        edges = list(G.edges())
        df = pd.DataFrame(edges, columns=['src', 'dst'])
        return df
    
    return generate_graph_nx, nx_to_edgelist


@app.cell
def __(nx, time, NX_AVAILABLE):
    """NetworkX algorithm implementations"""
    
    def benchmark_pagerank_nx(G) -> Tuple[float, dict]:
        """Benchmark PageRank on NetworkX"""
        if not NX_AVAILABLE:
            return None, None
        
        start = time.time()
        pr = nx.pagerank(G, alpha=0.85, max_iter=100)
        elapsed = time.time() - start
        return elapsed, pr
    
    def benchmark_louvain_nx(G) -> Tuple[float, dict]:
        """Benchmark Louvain community detection on NetworkX"""
        if not NX_AVAILABLE:
            return None, None
        
        try:
            import networkx.algorithms.community as nx_comm
            start = time.time()
            communities = nx_comm.louvain_communities(G)
            elapsed = time.time() - start
            
            # Convert to node->community dict
            node_community = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    node_community[node] = comm_id
            
            return elapsed, node_community
        except Exception as e:
            return None, None
    
    def benchmark_connected_components_nx(G) -> Tuple[float, int]:
        """Benchmark connected components on NetworkX"""
        if not NX_AVAILABLE:
            return None, None
        
        start = time.time()
        n_components = nx.number_connected_components(G)
        elapsed = time.time() - start
        return elapsed, n_components
    
    def benchmark_betweenness_nx(G, k: int = 100) -> Tuple[float, dict]:
        """Benchmark betweenness centrality on NetworkX (approximate)"""
        if not NX_AVAILABLE:
            return None, None
        
        start = time.time()
        bc = nx.betweenness_centrality(G, k=min(k, len(G.nodes())))
        elapsed = time.time() - start
        return elapsed, bc
    
    return (
        benchmark_pagerank_nx, benchmark_louvain_nx,
        benchmark_connected_components_nx, benchmark_betweenness_nx
    )


@app.cell
def __(cugraph, pd, time, CUGRAPH_AVAILABLE):
    """cuGraph algorithm implementations"""
    
    def benchmark_pagerank_cugraph(edgelist_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Benchmark PageRank on cuGraph"""
        if not CUGRAPH_AVAILABLE:
            return None, None
        
        # Create cuGraph
        import cudf
        edgelist_cudf = cudf.from_pandas(edgelist_df)
        G = cugraph.Graph()
        G.from_cudf_edgelist(edgelist_cudf, source='src', destination='dst')
        
        start = time.time()
        pr = cugraph.pagerank(G, alpha=0.85, max_iter=100)
        elapsed = time.time() - start
        
        return elapsed, pr.to_pandas()
    
    def benchmark_louvain_cugraph(edgelist_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Benchmark Louvain community detection on cuGraph"""
        if not CUGRAPH_AVAILABLE:
            return None, None
        
        import cudf
        edgelist_cudf = cudf.from_pandas(edgelist_df)
        G = cugraph.Graph()
        G.from_cudf_edgelist(edgelist_cudf, source='src', destination='dst')
        
        start = time.time()
        communities, modularity = cugraph.louvain(G)
        elapsed = time.time() - start
        
        return elapsed, communities.to_pandas()
    
    def benchmark_connected_components_cugraph(edgelist_df: pd.DataFrame) -> Tuple[float, int]:
        """Benchmark connected components on cuGraph"""
        if not CUGRAPH_AVAILABLE:
            return None, None
        
        import cudf
        edgelist_cudf = cudf.from_pandas(edgelist_df)
        G = cugraph.Graph()
        G.from_cudf_edgelist(edgelist_cudf, source='src', destination='dst')
        
        start = time.time()
        components = cugraph.connected_components(G)
        n_components = components['labels'].nunique()
        elapsed = time.time() - start
        
        return elapsed, n_components
    
    def benchmark_betweenness_cugraph(edgelist_df: pd.DataFrame, k: int = 100) -> Tuple[float, pd.DataFrame]:
        """Benchmark betweenness centrality on cuGraph"""
        if not CUGRAPH_AVAILABLE:
            return None, None
        
        import cudf
        edgelist_cudf = cudf.from_pandas(edgelist_df)
        G = cugraph.Graph()
        G.from_cudf_edgelist(edgelist_cudf, source='src', destination='dst')
        
        start = time.time()
        bc = cugraph.betweenness_centrality(G, k=k)
        elapsed = time.time() - start
        
        return elapsed, bc.to_pandas()
    
    return (
        benchmark_pagerank_cugraph, benchmark_louvain_cugraph,
        benchmark_connected_components_cugraph, benchmark_betweenness_cugraph
    )


@app.cell
def __(
    run_benchmark_btn, graph_type, num_nodes, edge_density, algorithms,
    generate_graph_nx, nx_to_edgelist, benchmark_pagerank_nx,
    benchmark_louvain_nx, benchmark_connected_components_nx,
    benchmark_betweenness_nx, benchmark_pagerank_cugraph,
    benchmark_louvain_cugraph, benchmark_connected_components_cugraph,
    benchmark_betweenness_cugraph, NX_AVAILABLE, CUGRAPH_AVAILABLE, mo
):
    """Run graph benchmarks"""
    
    graph_results = None
    
    if run_benchmark_btn.value and len(algorithms.value) > 0:
        mo.md("### 🔄 Running graph benchmarks...")
        
        try:
            n_nodes = 10 ** num_nodes.value
            avg_degree = edge_density.value
            
            # Generate graph
            mo.md(f"Generating {graph_type.value} graph with {n_nodes:,} nodes...")
            
            if NX_AVAILABLE:
                G = generate_graph_nx(graph_type.value, n_nodes, avg_degree)
                edgelist_df = nx_to_edgelist(G)
                actual_edges = len(edgelist_df)
            else:
                raise ImportError("NetworkX required for graph generation")
            
            # Run benchmarks
            results = {
                'algorithm': [],
                'nx_time': [],
                'cugraph_time': [],
                'speedup': [],
                'result_summary': []
            }
            
            for algo in algorithms.value:
                mo.md(f"Benchmarking {algo}...")
                
                # NetworkX benchmark
                if algo == 'pagerank':
                    nx_time, nx_result = benchmark_pagerank_nx(G)
                    if CUGRAPH_AVAILABLE:
                        cg_time, cg_result = benchmark_pagerank_cugraph(edgelist_df)
                    else:
                        cg_time, cg_result = None, None
                    result_summary = f"{len(nx_result)} nodes ranked" if nx_result else "N/A"
                
                elif algo == 'louvain':
                    nx_time, nx_result = benchmark_louvain_nx(G)
                    if CUGRAPH_AVAILABLE:
                        cg_time, cg_result = benchmark_louvain_cugraph(edgelist_df)
                    else:
                        cg_time, cg_result = None, None
                    
                    if nx_result:
                        n_communities = len(set(nx_result.values()))
                        result_summary = f"{n_communities} communities found"
                    else:
                        result_summary = "N/A"
                
                elif algo == 'connected_components':
                    nx_time, nx_result = benchmark_connected_components_nx(G)
                    if CUGRAPH_AVAILABLE:
                        cg_time, cg_result = benchmark_connected_components_cugraph(edgelist_df)
                    else:
                        cg_time, cg_result = None, None
                    result_summary = f"{nx_result} components" if nx_result else "N/A"
                
                elif algo == 'betweenness':
                    nx_time, nx_result = benchmark_betweenness_nx(G, k=100)
                    if CUGRAPH_AVAILABLE:
                        cg_time, cg_result = benchmark_betweenness_cugraph(edgelist_df, k=100)
                    else:
                        cg_time, cg_result = None, None
                    result_summary = f"{len(nx_result)} nodes analyzed" if nx_result else "N/A"
                
                else:
                    continue
                
                # Calculate speedup
                speedup = nx_time / cg_time if (nx_time and cg_time) else None
                
                results['algorithm'].append(algo.replace('_', ' ').title())
                results['nx_time'].append(nx_time)
                results['cugraph_time'].append(cg_time)
                results['speedup'].append(speedup)
                results['result_summary'].append(result_summary)
            
            graph_results = {
                'results': results,
                'graph_type': graph_type.value,
                'num_nodes': n_nodes,
                'num_edges': actual_edges,
                'avg_degree': actual_edges / n_nodes,
                'success': True
            }
            
        except Exception as e:
            graph_results = {
                'error': str(e),
                'success': False
            }
    
    return graph_results,


@app.cell
def __(graph_results, mo, go, CUGRAPH_AVAILABLE):
    """Visualize benchmark results"""
    
    if graph_results is None:
        mo.callout(
            mo.md("**Click 'Run Graph Benchmark' to start**"),
            kind="info"
        )
    elif not graph_results.get('success', False):
        mo.callout(
            mo.md(f"**Benchmark Error**: {graph_results.get('error', 'Unknown')}"),
            kind="danger"
        )
    else:
        results = graph_results['results']
        
        # Graph statistics
        graph_stats = {
            'Property': [
                'Graph Type',
                'Number of Nodes',
                'Number of Edges',
                'Average Degree',
                'Algorithms Tested'
            ],
            'Value': [
                graph_results['graph_type'].replace('_', ' ').title(),
                f"{graph_results['num_nodes']:,}",
                f"{graph_results['num_edges']:,}",
                f"{graph_results['avg_degree']:.2f}",
                str(len(results['algorithm']))
            ]
        }
        
        # Results table
        table_data = {
            'Algorithm': results['algorithm'],
            'Result': results['result_summary'],
            'NetworkX Time (s)': [f"{t:.4f}" if t else "N/A" for t in results['nx_time']],
        }
        
        if CUGRAPH_AVAILABLE:
            table_data['cuGraph Time (s)'] = [f"{t:.4f}" if t else "N/A" for t in results['cugraph_time']]
            table_data['Speedup'] = [f"{s:.2f}x" if s else "N/A" for s in results['speedup']]
        
        # Performance comparison chart
        if CUGRAPH_AVAILABLE:
            fig_perf = go.Figure()
            
            fig_perf.add_trace(go.Bar(
                name='NetworkX (CPU)',
                x=results['algorithm'],
                y=results['nx_time'],
                marker_color='#ff6b6b'
            ))
            
            fig_perf.add_trace(go.Bar(
                name='cuGraph (GPU)',
                x=results['algorithm'],
                y=results['cugraph_time'],
                marker_color='#51cf66'
            ))
            
            fig_perf.update_layout(
                title="Execution Time Comparison",
                xaxis_title="Algorithm",
                yaxis_title="Time (seconds)",
                barmode='group',
                height=400
            )
            
            # Speedup chart
            valid_speedups = [(algo, s) for algo, s in zip(results['algorithm'], results['speedup']) if s is not None]
            
            if valid_speedups:
                algos, speedups = zip(*valid_speedups)
                
                fig_speedup = go.Figure()
                
                fig_speedup.add_trace(go.Bar(
                    x=list(algos),
                    y=list(speedups),
                    marker_color='#4c6ef5',
                    text=[f"{s:.1f}x" for s in speedups],
                    textposition='outside'
                ))
                
                fig_speedup.update_layout(
                    title="GPU Speedup Factor",
                    xaxis_title="Algorithm",
                    yaxis_title="Speedup (higher is better)",
                    height=400
                )
                
                avg_speedup = sum(speedups) / len(speedups)
                
                mo.vstack([
                    mo.md("### ✅ Benchmark Complete!"),
                    mo.ui.table(graph_stats, label="Graph Statistics"),
                    mo.md("### 📊 Performance Results"),
                    mo.ui.table(table_data),
                    mo.md("### 📈 Performance Comparison"),
                    mo.hstack([
                        mo.ui.plotly(fig_perf),
                        mo.ui.plotly(fig_speedup)
                    ]),
                    mo.callout(
                        mo.md(f"""
                        **Performance Summary**:
                        - Graph: **{graph_results['num_nodes']:,}** nodes, **{graph_results['num_edges']:,}** edges
                        - Average GPU speedup: **{avg_speedup:.1f}x**
                        - cuGraph processes graphs entirely on GPU memory
                        - Larger graphs show even better speedups!
                        """),
                        kind="success"
                    )
                ])
            else:
                mo.vstack([
                    mo.md("### ✅ Benchmark Complete!"),
                    mo.ui.table(graph_stats, label="Graph Statistics"),
                    mo.ui.table(table_data),
                    mo.ui.plotly(fig_perf)
                ])
        else:
            mo.vstack([
                mo.md("### 📊 NetworkX Results (CPU only)"),
                mo.ui.table(graph_stats, label="Graph Statistics"),
                mo.ui.table(table_data),
                mo.callout(
                    mo.md("""
                    **Install RAPIDS cuGraph** to see GPU acceleration:
                    ```bash
                    conda install -c rapidsai -c conda-forge -c nvidia \
                        cugraph=24.04 python=3.10 cudatoolkit=11.8
                    ```
                    """),
                    kind="info"
                )
            ])
    return


@app.cell
def __(mo):
    """Documentation and resources"""
    mo.md(
        """
        ---
        
        ## 🎯 Graph Analytics with cuGraph
        
        **Supported Graph Types**:
        - **Scale-free**: Power-law degree distribution (social networks, web graphs)
        - **Random**: Erdős–Rényi model (baseline for comparison)
        - **Small-world**: High clustering + short paths (biological networks)
        - **Complete**: Every node connected to every other
        
        **Algorithm Complexity**:
        
        | Algorithm | NetworkX | cuGraph | Speedup |
        |-----------|----------|---------|---------|
        | PageRank | O(E·k) | O(E·k/p) | 50-100x |
        | Louvain | O(E·log V) | O(E·log V/p) | 30-80x |
        | BFS | O(V+E) | O((V+E)/p) | 100-200x |
        | Connected Components | O(V+E) | O((V+E)/p) | 150-300x |
        | Betweenness | O(V·E) | O(V·E/p) | 50-150x |
        
        *p = parallelism factor (GPU cores)*
        
        **Key Algorithms**:
        
        **1. PageRank** (link analysis):
        - Ranks nodes by importance
        - Used by Google Search
        - Iterative: converges to stationary distribution
        
        **2. Louvain** (community detection):
        - Finds clusters/communities
        - Maximizes modularity
        - Hierarchical: produces dendrogram
        
        **3. Connected Components**:
        - Finds disconnected subgraphs
        - Union-find on GPU
        - Useful for clustering
        
        **4. Betweenness Centrality**:
        - Identifies bridge nodes
        - Measures information flow
        - Expensive: O(V·E) complexity
        
        ### 🚀 Production Use Cases
        
        **Social Network Analysis**:
        ```python
        import cugraph
        import cudf
        
        # Load graph (billions of edges)
        edges = cudf.read_csv('social_network.csv')
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges, source='user1', destination='user2')
        
        # Find influencers
        pagerank = cugraph.pagerank(G)
        top_influencers = pagerank.nlargest(100, 'pagerank')
        
        # Detect communities
        communities, modularity = cugraph.louvain(G)
        print(f"Found {communities['partition'].nunique()} communities")
        print(f"Modularity: {modularity:.4f}")
        ```
        
        **Fraud Detection** (network analysis):
        ```python
        # Find suspicious connected components
        components = cugraph.connected_components(G)
        
        # Identify bridge accounts
        betweenness = cugraph.betweenness_centrality(G)
        suspicious = betweenness.nlargest(1000, 'betweenness_centrality')
        ```
        
        **Recommendation Systems**:
        ```python
        # Build user-item bipartite graph
        # Run PageRank for recommendations
        recommendations = cugraph.pagerank(G, personalization_vector=user_prefs)
        ```
        
        **Knowledge Graphs**:
        ```python
        # Multi-hop path finding
        paths = cugraph.bfs(G, start=entity_id, depth_limit=3)
        
        # Subgraph extraction
        subgraph = cugraph.subgraph(G, vertices=entity_list)
        ```
        
        ### 📊 Performance Optimization
        
        **Memory Management**:
        - Use CSR (Compressed Sparse Row) format for large graphs
        - Enable unified memory for graphs larger than GPU memory
        - Batch processing for very large graphs
        
        **Multi-GPU Scaling**:
        ```python
        from cugraph.dask import Graph
        import dask_cudf
        
        # Distribute graph across multiple GPUs
        edges_ddf = dask_cudf.read_csv('huge_graph.csv')
        G = Graph()
        G.from_dask_cudf_edgelist(edges_ddf)
        
        # Run algorithm across GPUs
        pagerank = cugraph.pagerank(G)
        ```
        
        **Run this notebook**:
        ```bash
        python graph_analytics_cugraph.py
        ```
        
        **Deploy as app**:
        ```bash
        marimo run graph_analytics_cugraph.py
        ```
        
        ### 📖 Resources
        - [RAPIDS cuGraph Documentation](https://docs.rapids.ai/api/cugraph/stable/)
        - [cuGraph GitHub](https://github.com/rapidsai/cugraph)
        - [Graph Algorithms Book](https://www.manning.com/books/graph-algorithms)
        - [NetworkX Documentation](https://networkx.org/)
        - [Brev.dev Platform](https://brev.dev)
        """
    )
    return


if __name__ == "__main__":
    app.run()

