import graph_tool.all as gt
import polars as pl
import numpy as np
import random
from collections import defaultdict
import os.path

print("Loading tracking data...")
players = pl.read_csv("nfl-big-data-bowl-2025/players.csv")
tracking = pl.read_csv("nfl-big-data-bowl-2025/tracking_week_1.csv", 
                      null_values=["NA", "na", "N/A", "n/a", "NULL", "null", "None", "none"])

pbp_file = "pbp_2022.csv"
if not os.path.isfile(pbp_file):
    print(f"NFL play-by-play data file '{pbp_file}' not found!")
    print("Please run the R script first: Rscript get_nfl_data.R")
    exit(1)

def group_rows_by_frame(data, gameId, playId):
    frames = defaultdict(list)
    for row in data.iter_rows():
        frame_id = row[4]
        if (frame_id % 24 == 0 or frame_id == 1) and row[0] == gameId and row[1] == playId:
            frames[frame_id].append(row)
    return frames

def construct_graph(data, gameId, playId):
    g = gt.Graph(directed=False)

    player_id_prop = g.new_vertex_property("string")
    x_coord_prop = g.new_vertex_property("float")
    y_coord_prop = g.new_vertex_property("float")
    
    g.vertex_properties["player_id"] = player_id_prop
    g.vertex_properties["x"] = x_coord_prop
    g.vertex_properties["y"] = y_coord_prop

    weight_prop = g.new_edge_property("float")
    g.edge_properties["weight"] = weight_prop

    playerIDs = set()
    vertex_dict = {}
    res = []
    graphs = []

    for row in data.iter_rows():
        if (row[0] == gameId and 
            row[1] == playId and 
            row[3] != 'football' and  
            ((int(row[4]) % 24 == 0) or 
             (int(row[4]) == 1))):

            res.append(row)
            
            if row[2] != 'None' and row[2] not in vertex_dict:
                playerIDs.add(row[2])
                v = g.add_vertex()
                player_id_prop[v] = row[2]
                vertex_dict[row[2]] = v 

    frame_groups = group_rows_by_frame(data, gameId, playId)
    
    for frame, rows in frame_groups.items():
        for row in rows:
            nfl_id = row[2]
            if nfl_id in vertex_dict:
                v = vertex_dict[nfl_id]
                x_coord_prop[v] = float(row[10])
                y_coord_prop[v] = float(row[11])
        
        g.clear_edges()
        vertices = list(vertex_dict.values())
        n = len(vertices)
        print(f"Number of vertices: {n}")
        for i in range(n):
            for j in range(i + 1, n):
                v1 = vertices[i]
                v2 = vertices[j]
                dx = x_coord_prop[v1] - x_coord_prop[v2]
                dy = y_coord_prop[v1] - y_coord_prop[v2]
                dist = (dx**2 + dy**2)**0.5
                weight = 1 / dist if dist != 0 else 0
                e = g.add_edge(v1, v2)
                weight_prop[e] = weight

        print(f"Frame: {frame} Edges computed: {g.num_edges()}")
        graphs.append(g.copy())

    return graphs


print("Getting a random play...")
unique_plays = tracking.select(['gameId', 'playId']).unique()
random_idx = random.randint(0, len(unique_plays) - 1)
random_game_id = unique_plays[random_idx, 0]
random_play_id = unique_plays[random_idx, 1]

print(f"Analyzing GameID: {random_game_id}, PlayID: {random_play_id}")

# Create graph 
play_graphs = construct_graph(tracking, random_game_id, random_play_id)
print(f"Created {len(play_graphs)} frame graphs for this play")

print("Loading play-by-play data from CSV...")
pbp_data = pl.read_csv(
    pbp_file, 
    null_values=["NA", "na", "N/A", "n/a", "NULL", "null", "None", "none"],
    infer_schema_length=10000,
    schema_overrides={"total_line": pl.Float64}
)

play_data = pbp_data.filter(pl.col('play_id') == random_play_id)

if len(play_data) > 0:
    if len(play_data) > 1:
        str_game_id = str(random_game_id)
        game_year = str_game_id[0:4]  
        game_month = str_game_id[4:6]  
        game_day = str_game_id[6:8]  
        
        print(f"Looking for games on date: {game_year}-{game_month}-{game_day}")
        
        print(f"Found {len(play_data)} potential play matches. Trying to find exact match...")
        possible_matches = []
        
        for row in play_data.iter_rows():
            game_id_col = row[play_data.columns.index('game_id')]
            play_id_col = row[play_data.columns.index('play_id')]
            game_date_col = row[play_data.columns.index('game_date')] if 'game_date' in play_data.columns else "unknown"
            
            print(f"Potential match: game_id={game_id_col}, play_id={play_id_col}, date={game_date_col}")
            
            if 'game_date' in play_data.columns:
                if game_date_col and f"{game_year}-{game_month}-{game_day}" in str(game_date_col):
                    print(f"Date match found: {game_date_col}")
                    possible_matches.append(row)
            else:
                possible_matches.append(row)
        
        if len(possible_matches) == 1:
            play_data = pl.DataFrame([possible_matches[0]], schema=play_data.schema)
        elif len(possible_matches) > 0:
            print(f"Multiple matches found ({len(possible_matches)}), using the first one for analysis")
            play_data = pl.DataFrame([possible_matches[0]], schema=play_data.schema)
        else:
            print("No matches found after date filtering, using the first original match")
            first_row = next(play_data.iter_rows())
            play_data = pl.DataFrame([first_row], schema=play_data.schema)
    
    epa = play_data.select('epa').item()
    def_epa = -epa  
    play_desc = play_data.select('desc').item() if 'desc' in play_data.columns else play_data.select('play_description').item() if 'play_description' in play_data.columns else "N/A"
    print(f"Play description: {play_desc}")
    print(f"Offensive EPA: {epa}")
    print(f"Defensive EPA: {def_epa}")
else:
    print("No matching play found in play-by-play data")

# Compute network metrics for each frame graph
frame_metrics = []

for i, g in enumerate(play_graphs):
    print(f"\nAnalyzing Frame Graph {i+1}:")
    
    weight_prop = g.edge_properties["weight"]
    
    # Weighted Betweenness Centrality
    try:
        vertex_btw, edge_btw = gt.betweenness(g, weight=weight_prop)
        avg_betweenness = vertex_btw.a.mean()
        max_betweenness = vertex_btw.a.max() if len(vertex_btw.a) > 0 else 0
        print(f"Weighted Betweenness Centrality - Avg: {avg_betweenness:.4f}, Max: {max_betweenness:.4f}")
    except Exception as e:
        print(f"Error calculating weighted betweenness centrality: {e}")
        avg_betweenness = 0.0
        max_betweenness = 0.0
    
    # Weighted Closeness Centrality
    try:
        closeness = gt.closeness(g, weight=weight_prop)
        avg_closeness = closeness.a.mean()
        print(f"Weighted Closeness Centrality: {avg_closeness:.4f}")
    except Exception as e:
        print(f"Error calculating weighted closeness centrality: {e}")
        avg_closeness = 0.0
    
    # Weighted Eigenvector Centrality
    try:
        eigen = gt.eigenvector(g, weight=weight_prop)
        if isinstance(eigen, tuple):
            eigen_vector = eigen[1]
        else:
            eigen_vector = eigen
            
        if hasattr(eigen_vector, 'a'):
            avg_eigen = eigen_vector.a.mean()
        else:
            avg_eigen = sum(eigen_vector[v] for v in g.vertices()) / g.num_vertices()
            
        print(f"Weighted Eigenvector Centrality: {avg_eigen:.4f}")
    except Exception as e:
        print(f"Error calculating weighted eigenvector centrality: {e}")
        avg_eigen = 0.0
    
    # Weighted PageRank 
    try:
        pagerank = gt.pagerank(g, weight=weight_prop)
        avg_pagerank = pagerank.a.mean()
        print(f"Weighted PageRank: {avg_pagerank:.4f}")
    except Exception as e:
        print(f"Error calculating weighted PageRank: {e}")
        avg_pagerank = 0.0
    
    # 5. Weighted Katz Centrality 
    try:
        katz = gt.katz(g, weight=weight_prop)
        avg_katz = katz.a.mean()
        print(f"Weighted Katz Centrality: {avg_katz:.4f}")
    except Exception as e:
        print(f"Error calculating weighted Katz centrality: {e}")
        avg_katz = 0.0
        
    
    # Weighted Clustering Coefficient
    try:
        clustering_global = gt.global_clustering(g, weight=weight_prop)[0]
        clustering_local = gt.local_clustering(g, weight=weight_prop)
        avg_local_clustering = clustering_local.a.mean()
        print(f"Weighted Global Clustering Coefficient: {clustering_global:.4f}")
        print(f"Weighted Local Clustering Coefficient: {avg_local_clustering:.4f}")
    except Exception as e:
        print(f"Error calculating weighted clustering: {e}")
        clustering_global = 0.0
        avg_local_clustering = 0.0
    
    # Network Density (Always the same)
    n_vertices = g.num_vertices()
    n_edges = g.num_edges()
    max_edges = n_vertices * (n_vertices - 1) / 2  # For undirected graph
    density = n_edges / max_edges if max_edges > 0 else 0
    print(f"Network Density: {density:.4f}")
    
    # Average Edge Weight
    total_weight = sum(weight_prop[e] for e in g.edges())
    avg_weight = total_weight / n_edges if n_edges > 0 else 0
    print(f"Average Edge Weight: {avg_weight:.4f}")
    
    # 7. Strength Distribution (weighted degree)
    vertex_strength = g.new_vertex_property("float")
    for v in g.vertices():
        strength = sum(weight_prop[e] for e in v.out_edges())
        vertex_strength[v] = strength
    
    avg_strength = sum(vertex_strength[v] for v in g.vertices()) / n_vertices if n_vertices > 0 else 0
    max_strength = max(vertex_strength[v] for v in g.vertices()) if n_vertices > 0 else 0
    print(f"Average Node Strength: {avg_strength:.4f}")
    print(f"Maximum Node Strength: {max_strength:.4f}")
    
    # 8. Robustness - simulating removal based on weighted centrality
    btw_ranking = np.argsort(vertex_btw.a)[::-1]  
    
    g_copy = g.copy()
    nodes_to_remove = btw_ranking[:int(n_vertices * 0.2)]
    
    for v in sorted(nodes_to_remove, reverse=True):  
        g_copy.remove_vertex(v)
    
    comp, hist = gt.label_components(g_copy)
    num_components = len(hist)
    largest_comp_size = max(hist) if len(hist) > 0 else 0
    robustness = largest_comp_size / g_copy.num_vertices() if g_copy.num_vertices() > 0 else 0
    
    print(f"After removing top 20% central nodes:")
    print(f"  Number of components: {num_components}")
    print(f"  Largest component size: {largest_comp_size}")
    print(f"  Robustness (size of largest component / total nodes): {robustness:.4f}")
    
    # 9. Weight-based Distance Analysis
    all_weights = [weight_prop[e] for e in g.edges()]
    min_weight = min(all_weights) if all_weights else 0
    max_weight = max(all_weights) if all_weights else 0
    print(f"Minimum Edge Weight: {min_weight:.4f}")
    print(f"Maximum Edge Weight: {max_weight:.4f}")
    
    # Store metrics for this frame
    frame_metrics.append({
        'frame': i+1,
        'avg_betweenness': avg_betweenness,
        'max_betweenness': max_betweenness,
        'avg_closeness': avg_closeness,
        'avg_eigenvector': avg_eigen,
        'avg_pagerank': avg_pagerank,
        'avg_katz': avg_katz,
        'global_clustering': clustering_global,
        'avg_local_clustering': avg_local_clustering,
        'density': density,
        'avg_weight': avg_weight,
        'avg_strength': avg_strength,
        'max_strength': max_strength,
        'robustness': robustness,
        'min_weight': min_weight,
        'max_weight': max_weight
    })

# Print summary of metrics across all frames
print("\nSummary of Weighted Network Metrics Across All Frames:")
metrics_df = pl.DataFrame(frame_metrics)
summary = metrics_df.select(
    pl.col('avg_betweenness').mean().alias('avg_betweenness'),
    pl.col('max_betweenness').mean().alias('max_betweenness'),
    pl.col('avg_closeness').mean().alias('avg_closeness'),
    pl.col('avg_eigenvector').mean().alias('avg_eigenvector'),
    pl.col('avg_pagerank').mean().alias('avg_pagerank'),
    pl.col('avg_katz').mean().alias('avg_katz'),
    pl.col('global_clustering').mean().alias('global_clustering'),
    pl.col('avg_local_clustering').mean().alias('avg_local_clustering'),
    pl.col('density').mean().alias('density'),
    pl.col('avg_weight').mean().alias('avg_weight'),
    pl.col('avg_strength').mean().alias('avg_strength'),
    pl.col('max_strength').mean().alias('max_strength'),
    pl.col('robustness').mean().alias('robustness'),
    pl.col('min_weight').mean().alias('min_weight'),
    pl.col('max_weight').mean().alias('max_weight')
)
print(summary) 