"""圆形障碍重叠率实验：4个等级×5张地图×10个搜索种子×3种算法。"""
import argparse, csv, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiment_rrt_star import run_once

OVERLAPS = (0, 20, 40, 60)
METHODS = ("RRT*", "GoalBias", "Hybrid")
COLORS = {"RRT*": "#9AA0A6", "GoalBias": "#4C78A8", "Hybrid": "#E07A5F"}
METRICS = ("planning_time", "first_solution_iteration", "node_count", "path_length")

def trial(task):
    overlap, map_seed, search_seed, method, max_iterations = task
    metrics, *_ = run_once(method, search_seed, env_type=f"overlap_{overlap}_{map_seed}",
                           environment_path=None, rectangle_length=30.0,
                           rectangle_width=20.0, allow_reverse=False,
                           max_iterations=max_iterations)
    row = {"overlap": overlap, "map_seed": map_seed, "search_seed": search_seed, "method": method}
    row.update(metrics)
    return row

def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

def summarize(rows):
    out=[]
    for overlap in OVERLAPS:
        for method in METHODS:
            group=[r for r in rows if int(r["overlap"])==overlap and r["method"]==method]
            item={"overlap":overlap,"method":method,"trials":len(group),
                  "success_rate_percent":100*sum(int(r.get("success",0)) for r in group)/len(group)}
            for metric in METRICS:
                v=[float(r[metric]) for r in group if int(r.get("success",0))]
                item[f"mean_{metric}"]=statistics.mean(v) if v else float("nan")
                item[f"std_{metric}"]=statistics.stdev(v) if len(v)>1 else float("nan")
                item[f"median_{metric}"]=statistics.median(v) if v else float("nan")
            out.append(item)
    return out

def draw(summary, stem):
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
                         "font.size":8,"figure.facecolor":"white","axes.facecolor":"white",
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    labels={"planning_time":"Planning time (s)","first_solution_iteration":"First-solution iteration",
            "node_count":"Number of nodes","path_length":"Path length (m)"}
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)
    for ax,metric in zip(axes.flat,METRICS):
        for method in METHODS:
            g=[r for r in summary if r["method"]==method]
            ax.errorbar([r["overlap"] for r in g],[r[f"mean_{metric}"] for r in g],
                        yerr=[r[f"std_{metric}"] for r in g],marker="o",capsize=3,
                        linewidth=1.5,color=COLORS[method],label=method)
        ax.set_xlabel("Obstacle overlap (%)"); ax.set_ylabel(labels[metric])
        ax.grid(axis="y",color="#D9D9D9",linewidth=.6,alpha=.7)
    axes[0,0].legend(frameon=False,ncol=3,loc="upper left")
    fig.suptitle("Overlap sensitivity of orchard path planners",fontsize=10)
    for ext,kwargs in (("png",{"dpi":300}),("pdf",{}),("svg",{})):
        fig.savefig(stem+"."+ext,bbox_inches="tight",facecolor="white",**kwargs)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--max-iterations",type=int,default=1500); args=parser.parse_args()
    out=os.path.join(os.path.dirname(__file__),"overlap_results"); os.makedirs(out,exist_ok=True)
    fields=["overlap","map_seed","search_seed","method","success",*METRICS]
    checkpoint=os.path.join(out,"overlap_detail_checkpoint.csv"); rows=[]
    if os.path.exists(checkpoint):
        with open(checkpoint,encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    done={(int(r["overlap"]),int(r["map_seed"]),int(r["search_seed"]),r["method"]) for r in rows}
    tasks=[(o,m,s,method,args.max_iterations) for o in OVERLAPS for m in range(5) for s in range(10) for method in METHODS
           if (o,m,s,method) not in done]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(trial,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result(); rows.append(row); write_csv(checkpoint,rows,fields)
            print(f"[{len(done)+i}/600] overlap={row['overlap']} map={row['map_seed']} search={row['search_seed']} {row['method']} success={row.get('success',0)}",flush=True)
    rows.sort(key=lambda r:(int(r["overlap"]),int(r["map_seed"]),int(r["search_seed"]),METHODS.index(r["method"])))
    summary=summarize(rows); write_csv(os.path.join(out,"overlap_detail.csv"),rows,fields)
    write_csv(os.path.join(out,"overlap_summary.csv"),summary,list(summary[0]))
    draw(summary,os.path.join(out,"overlap_sensitivity"))
    print("结果目录:",out)

if __name__ == "__main__": main()
