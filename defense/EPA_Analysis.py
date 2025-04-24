import graph_tool.all as gt
import polars as pl
import numpy as np
import os
from collections import defaultdict
import scipy.stats as st
from tqdm import tqdm

# all weeks config
WEEKS = range(1, 10)
PLAYERS_FILE = "nfl-big-data-bowl-2025/players.csv"
PBP_FILE     = "pbp_2022.csv"

GOOD_THRESHOLD = 0.25
BAD_THRESHOLD  = -0.25

# preload PBP & EPA lookup
pbp = (
    pl.read_csv(
       PBP_FILE,
       null_values=["NA","na","N/A","n/a","NULL","null","None","none"],
       infer_schema_length=10000,
       schema_overrides={"total_line": pl.Float64}
    )
    .with_columns([
        (-pl.col("epa")).alias("def_epa"),
        pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d")
    ])
)

epa_lookup = (
    pbp
    .select(["play_id", "game_date", "def_epa"])
    .unique()
    .rename({"play_id": "playId"})
)

# helpers
def group_rows_by_frame(rows, gameId, playId):
    frames = defaultdict(list)
    for r in rows:
        fid = r[4]
        if (fid % 24 == 0 or fid == 1) and r[0] == gameId and r[1] == playId:
            frames[fid].append(r)
    return frames

def construct_graph(rows, gameId, playId):
    g = gt.Graph(directed=False)
    pid_prop = g.new_vertex_property("string")
    x_prop   = g.new_vertex_property("float")
    y_prop   = g.new_vertex_property("float")
    g.vertex_properties["player_id"] = pid_prop
    g.vertex_properties["x"]         = x_prop
    g.vertex_properties["y"]         = y_prop
    w_prop   = g.new_edge_property("float")
    g.edge_properties["weight"] = w_prop

    vertex_dict = {}
    for r in rows:
        pid = r[2]
        if pid != "None" and pid not in vertex_dict:
            v = g.add_vertex()
            pid_prop[v] = pid
            vertex_dict[pid] = v

    for fr_rows in group_rows_by_frame(rows, gameId, playId).values():
        for r in fr_rows:
            pid = r[2]
            if pid in vertex_dict:
                v = vertex_dict[pid]
                x_prop[v] = float(r[10])
                y_prop[v] = float(r[11])
        g.clear_edges()
        verts = list(vertex_dict.values())
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                v1, v2 = verts[i], verts[j]
                dx = x_prop[v1] - x_prop[v2]
                dy = y_prop[v1] - y_prop[v2]
                dist = (dx*dx + dy*dy)**0.5
                w = 1/dist if dist > 0 else 0
                e = g.add_edge(v1, v2)
                w_prop[e] = w
        yield g.copy()

def compute_metrics(g):
    w = g.edge_properties["weight"]
    vb, _ = gt.betweenness(g, weight=w)
    cl    = gt.closeness(g, weight=w)
    eig   = gt.eigenvector(g, weight=w)
    eigv  = eig[1] if isinstance(eig, tuple) else eig
    pr    = gt.pagerank(g, weight=w)
    kz    = gt.katz(g, weight=w)
    gc    = gt.global_clustering(g, weight=w)[0]
    lc    = gt.local_clustering(g, weight=w)
    nV, nE = g.num_vertices(), g.num_edges()
    density = nE / (nV*(nV-1)/2) if nV>1 else 0
    avg_w = sum(w[e] for e in g.edges()) / nE if nE>0 else 0

    strength = g.new_vertex_property("float")
    for v in g.vertices():
        strength[v] = sum(w[e] for e in v.out_edges())
    avg_s = strength.a.mean()
    max_s = strength.a.max()

    rank = np.argsort(vb.a)[::-1]
    g2   = g.copy()
    for v_id in sorted(rank[:int(nV*0.2)], reverse=True):
        g2.remove_vertex(v_id)
    _, hist = gt.label_components(g2)
    rob = max(hist) / g2.num_vertices() if g2.num_vertices()>0 else 0

    return {
      "avg_betweenness": vb.a.mean(),
      "max_betweenness": vb.a.max(),
      "avg_closeness":   cl.a.mean(),
      "avg_eigen":       eigv.a.mean(),
      "avg_pagerank":    pr.a.mean(),
      "avg_katz":        kz.a.mean(),
      "global_clustering": gc,
      "avg_local_clustering": lc.a.mean(),
      "density": density,
      "avg_weight": avg_w,
      "avg_strength": avg_s,
      "max_strength": max_s,
      "robustness": rob
    }

def analyze_batch(df):
    results = []
    pairs = list(df.select(["gameId","playId"]).unique().iter_rows())
    for g_id, p_id in tqdm(pairs, total=len(pairs), desc="Analyzing plays", unit="plays"):
        grp = df.filter((pl.col("gameId")==g_id)&(pl.col("playId")==p_id))
        metrics = [compute_metrics(gx) for gx in construct_graph(list(grp.iter_rows()), g_id, p_id)]
        avg   = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        std   = {f"{k}_std": np.std([m[k] for m in metrics]) for k in metrics[0]}
        results.append({"gameId": g_id, "playId": p_id, **avg, **std})
    return pl.DataFrame(results)

# loop (weeks 1-9)
all_good = []
all_avg  = []
all_bad  = []
all_sum  = []

for wk in WEEKS:
    print(f"\n=== WEEK {wk} ===")
    fn = f"nfl-big-data-bowl-2025/tracking_week_{wk}.csv"
    tracking = pl.read_csv(fn, null_values=["NA","na","N/A","n/a","NULL","null","None","none"])
    tracking = tracking.with_columns([
        pl.col("gameId").cast(pl.Utf8).str.slice(0,8).str.strptime(pl.Date, "%Y%m%d").alias("game_date")
    ])

    tracks = tracking.join(epa_lookup, on=["playId","game_date"])
    bad     = tracks.filter(pl.col("def_epa") <= BAD_THRESHOLD)
    avg     = tracks.filter((pl.col("def_epa") > BAD_THRESHOLD) & (pl.col("def_epa") < GOOD_THRESHOLD))
    good    = tracks.filter(pl.col("def_epa") >= GOOD_THRESHOLD)

    bad_met  = analyze_batch(bad).with_columns([pl.lit("bad").alias("group"),
                                                pl.lit(wk).alias("week")])
    avg_met  = analyze_batch(avg).with_columns([pl.lit("average").alias("group"),
                                                pl.lit(wk).alias("week")])
    good_met = analyze_batch(good).with_columns([pl.lit("good").alias("group"),
                                                 pl.lit(wk).alias("week")])

    # week-by-week t-tests between all three groups 
    summary = []
    metric_cols = [c for c in good_met.columns 
                   if not c.endswith("_std") and c not in ["gameId","playId","week","group"]]
    for m in metric_cols:
        t_gb, p_gb = st.ttest_ind(good_met[m].to_numpy(), bad_met[m].to_numpy(), equal_var=False)
        t_ga, p_ga = st.ttest_ind(good_met[m].to_numpy(), avg_met[m].to_numpy(), equal_var=False)
        t_ab, p_ab = st.ttest_ind(avg_met[m].to_numpy(), bad_met[m].to_numpy(), equal_var=False)
        summary.append({
            "week": wk,
            "metric": m,
            "good_vs_bad_t":   t_gb, "good_vs_bad_p":   p_gb,
            "good_vs_avg_t":   t_ga, "good_vs_avg_p":   p_ga,
            "avg_vs_bad_t":    t_ab, "avg_vs_bad_p":    p_ab,
        })
    sum_df = pl.DataFrame(summary)

    all_bad.append(bad_met)
    all_avg.append(avg_met)
    all_good.append(good_met)
    all_sum.append(sum_df)

# merge everything
bad_all  = pl.concat(all_bad)
avg_all  = pl.concat(all_avg)
good_all = pl.concat(all_good)
sum_all  = pl.concat(all_sum)

# save
bad_all.write_csv("bad_weeks1-9_metrics.csv")
avg_all.write_csv("average_weeks1-9_metrics.csv")
good_all.write_csv("good_weeks1-9_metrics.csv")
sum_all.write_csv("summary_weeks1-9_three_cohorts.csv")

print("\nDone! Three‐cohort results written to CSV.")
