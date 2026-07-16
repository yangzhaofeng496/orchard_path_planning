"""根据三算法30种子结果生成PDF分析报告。"""
import csv
import os
import sys
from statistics import mean, median

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


ROOT = os.path.dirname(__file__)
RESULTS = os.path.join(ROOT, "three_planner_results")
OUT = os.path.join(ROOT, "three_planner_results", "three_planner_experiment_report.pdf")
FIGURE = os.path.join(RESULTS, "three_planners_comparison.png")


def load_rows():
    with open(os.path.join(RESULTS, "three_planners_detail.csv"), encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def vals(rows, scene, method, metric):
    return [float(r[metric]) for r in rows if r["scenario"] == scene and r["method"] == method]


def build_pdf():
    pdfmetrics.registerFont(TTFont("ArialUnicode", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", parent=styles["Title"], fontName="ArialUnicode", fontSize=18, leading=23, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="CNHeading", parent=styles["Heading2"], fontName="ArialUnicode", fontSize=12, leading=16, spaceBefore=9, spaceAfter=6))
    styles.add(ParagraphStyle(name="CNBody", parent=styles["BodyText"], fontName="ArialUnicode", fontSize=9.5, leading=15, spaceAfter=6))
    styles.add(ParagraphStyle(name="CNSmall", parent=styles["BodyText"], fontName="ArialUnicode", fontSize=8, leading=11, textColor=colors.HexColor("#555555")))

    rows = load_rows()
    scenes = [
        ("hybrid_single_blocker", "单圆封锁"),
        ("hybrid_staggered_trees", "交错果树"),
        ("hybrid_row_corridor", "行间通道"),
    ]
    methods = ["RRT*", "GoalBias", "Hybrid"]
    doc = SimpleDocTemplate(OUT, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=15*mm, bottomMargin=15*mm)
    story = [
        Paragraph("三种全局路径规划方法实验分析报告", styles["CNTitle"]),
        Paragraph("实验范围：3个圆形果园场景 × 3种规划器 × 30个随机种子，共270次单进程实验。所有组合成功率均为100%。", styles["CNBody"]),
        Paragraph("一、总体结论", styles["CNHeading"]),
        Paragraph("Hybrid在单圆封锁和交错果树场景中表现出稳定的搜索效率优势：相比RRT*和GoalBias，首次解迭代次数和有效节点数明显减少。行间通道中，Hybrid的中位运行时间较低，但少数种子出现较慢搜索，说明椭圆障碍簇引导需要进一步加入触发条件或缓存机制。", styles["CNBody"]),
        Paragraph("二、平均性能", styles["CNHeading"]),
    ]
    data = [["场景", "方法", "时间(s)", "首次解", "节点数", "路径(m)"]]
    for scene, scene_cn in scenes:
        for method in methods:
            line = [r for r in rows if r["scenario"] == scene and r["method"] == method]
            data.append([scene_cn, method, f"{mean(float(r['planning_time']) for r in line):.2f}", f"{mean(float(r['first_solution_iteration']) for r in line):.1f}", f"{mean(float(r['node_count']) for r in line):.1f}", f"{mean(float(r['path_length']) for r in line):.2f}"])
    table = Table(data, colWidths=[28*mm, 23*mm, 22*mm, 22*mm, 22*mm, 22*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "ArialUnicode"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#AAAAAA")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 8), Paragraph("三、结果解释", styles["CNHeading"])]
    story += [Paragraph("1）单圆封锁：Hybrid平均时间为0.72 s，GoalBias为1.47 s，RRT*为3.56 s。Hybrid在27/30个种子上快于GoalBias，说明目标方向被单个圆障碍阻挡时，切向子目标能够减少无效目标采样。", styles["CNBody"])]
    story += [Paragraph("2）交错果树：Hybrid平均时间为1.10 s，路径长度为135.06 m，均优于GoalBias和RRT*。这是最能支撑障碍簇椭圆化与切向引导有效性的场景。", styles["CNBody"])]
    story += [Paragraph("3）行间通道：Hybrid平均时间为7.62 s，高于GoalBias的5.26 s，但首次解迭代、节点数和路径长度略优。该场景中规则通道已经适合GoalBias，复杂椭圆更新的额外开销抵消了采样收益。", styles["CNBody"])]
    story += [Paragraph("四、局限与后续实验", styles["CNHeading"]), Paragraph("本报告覆盖的是三类典型场景对比。密度梯度、障碍重叠率、模块消融、矩形宽度和切向延伸距离等实验尚未纳入本次270次结果，不能将当前报告表述为完整消融结论。后续应优先完成：障碍重叠率实验、五版本消融实验，以及多随机地图实验。", styles["CNBody"])]
    story += [Paragraph("五、图形结果", styles["CNHeading"]), Image(FIGURE, width=178*mm, height=138*mm), Paragraph("图中箱线图表示30个随机种子的分布，散点表示单次实验结果；灰色、蓝色、橙色分别代表RRT*、GoalBias和Hybrid。", styles["CNSmall"])]
    doc.build(story)
    return OUT


if __name__ == "__main__":
    print(build_pdf())
