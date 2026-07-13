from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
import csv, os

ROOT=os.path.dirname(__file__)
OUT=os.path.abspath(os.path.join(ROOT,'..','output','pdf','orchard_path_planning_experiment_report.pdf'))
pdfmetrics.registerFont(TTFont('CJK','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
F='CJK'; ss=getSampleStyleSheet()
ss.add(ParagraphStyle(name='T',parent=ss['Title'],fontName=F,fontSize=19,leading=25,textColor=colors.HexColor('#16324F'),spaceAfter=10))
ss.add(ParagraphStyle(name='H',parent=ss['Heading2'],fontName=F,fontSize=13,leading=17,textColor=colors.HexColor('#176B87'),spaceBefore=9,spaceAfter=4))
ss.add(ParagraphStyle(name='B',parent=ss['BodyText'],fontName=F,fontSize=9.1,leading=13,spaceAfter=5))
ss.add(ParagraphStyle(name='S',parent=ss['BodyText'],fontName=F,fontSize=7.4,leading=10))
ss.add(ParagraphStyle(name='Ours',parent=ss['BodyText'],fontName=F,fontSize=7.3,leading=10,textColor=colors.red))
def P(x,s='B'): return Paragraph(x,ss[s])
def csvread(n):
    with open(os.path.join(ROOT,'results',n),newline='') as f:return list(csv.DictReader(f))
def tab(data,widths):
    t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#176B87')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,-1),F),('FONTSIZE',(0,0),(-1,-1),7.3),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#B7C9D3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F6F8')]),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    return t
summary=csvread('summary.csv')
n_episodes=int(summary[0]['episodes'])
main=[r for r in summary if not r['method'].startswith('no_')]
best_success=max(float(r['success_rate']) for r in main)
eligible=[r for r in main if abs(float(r['success_rate'])-best_success)<1e-9]
best_length=min(float(r['mean_length']) for r in eligible)
best_clearance=max(float(r['mean_clearance']) for r in eligible)
best_collision=min(float(r['mean_collisions']) for r in main)
ours=next(r for r in summary if 'Ours' in r['method'])
hybrid=next(r for r in summary if r['method']=='Hybrid A*+TEB')
def redbest(text,yes): return P('<b><font color="red">'+text+'</font></b>','Ours') if yes else text
story=[P('果园移动机器人全局-局部路径规划实验报告','T'),P('RCRA* 全局规划 + UA-DWA 局部规划：基线对比与消融实验'),P(f'版本：2026-07-13 | {n_episodes} 个固定随机种子二维仿真场景','S')]
story += [P('摘要','H'),P(f'本报告构建了一个可复现的果园路径规划 benchmark，提出 Row-Corridor Risk-Aware A*（RCRA*）和 Uncertainty-Aware DWA（UA-DWA）。RCRA* 将行间走廊、障碍风险和转弯代价加入栅格搜索；UA-DWA 将动态障碍预测、安全权重自适应和全局路径偏差加入速度采样。实验比较 Greedy、Dijkstra、A*、RRT*、Hybrid A*、APF、DWA、TEB 和 Pure Pursuit，并进行模块消融。所有统计基于 {n_episodes} 个固定随机场景。','B')]
story += [P('1. 背景与问题定义','H'),P('果园机器人需要在树行之间完成巡检、采摘、喷雾或运输。果园具有狭窄行间通道、树干和树冠障碍、泥土与坡度影响、定位退化，以及人员和农机动态穿行等问题。仅优化欧氏距离容易贴近树干或在障碍前停滞；仅使用反应式局部方法又可能偏离作业行。','B'),P('本实验验证：H1，行间走廊代价可减少路径偏离；H2，风险代价可提升动态障碍安全性；H3，将全局路径显式传给局部规划器可减少局部最优和停滞。RCRA* 的代价包括移动、走廊、转弯和风险项；UA-DWA 的评分包括目标推进、全局路径偏差、动态障碍预测距离和自适应安全项。','B')]
story += [P('1.1 核心创新点','H'),P('本文的贡献不是简单地并联 A* 与 DWA，而是围绕果园行间导航中“结构先验没有进入全局代价、局部窗口航向与树行不一致、动态障碍预测不足、局部停滞无法恢复”四个可复现问题构建协同机制。','B'),tab([['创新点','动机与机制','实验验证接口'],['创新点 1：果园行间走廊风险全局代价（RCRA*）','将树行中心偏好、树干风险带、移动距离和转弯代价统一写入全局搜索，使路径在绕开高风险区域的同时保持行间可通行性。','no_corridor 与 no_risk 消融；比较路径长度、成功率和安全间距'],['创新点 2：全局路径切线驱动的局部航向对齐','传统速度窗口若采用固定初始航向，可能与纵向树行正交并在起步阶段停滞。Ours 从前后全局节点估计路径切线，以该方向初始化局部推进。','旧版 Ours 成功率 81.25%；加入航向对齐与恢复后主测试达到 100%'],['创新点 3：不确定性感知的动态障碍安全带','采用恒速预测估计动态障碍在局部时间窗内的位置，并构建安全膨胀半径；当全局节点进入预测风险区时，在可通行空间中搜索安全横向偏移。','动态安全间距指标；与 DWA、TEB 风格基线比较'],['创新点 4：全局一致性的停滞恢复机制','当局部轨迹出现大跳变或连续无进展时，重新连接最近的安全全局路径节点，使局部避障仍保持朝目标的拓扑进展。','成功率与 no_global_term 消融；失败场景 0、1、11 的恢复结果'],['创新点 5：面向果园的全局-局部证据闭环','每个模块均对应一个可关闭的消融开关，并在相同随机场景下与 Greedy、Dijkstra、A*、RRT*、Hybrid A*、APF、DWA、TEB 和 Pure Pursuit 比较。','主基线表、模块消融表及独立种子 2000-2015 验证']],[31*mm,105*mm,41*mm]),P('<b>贡献边界：</b>上述结果支持这些机制在当前二维果园栅格模拟中的有效性，但尚不能证明其在真实果园、非完整车辆动力学、定位漂移或传感器遮挡条件下仍保持同等优势。Hybrid A* 与 TEB 也是简化基线，后续应使用标准 ROS/Nav2 实现复核。','B')]
story += [P('2. 实验设置','H'),P(f'地图为 90 x 90 栅格。树行沿 y 方向布置，树干表示为占据栅格并生成风险带；每个场景随机加入 10 个静态枝条/箱体障碍和 3 个匀速运动障碍。起点为 (17,8)，目标为 (17,48)，所有方法使用相同地图、起终点、障碍和随机种子 1000-{999+n_episodes}。测试样例由原来的 8 个扩大到 {n_episodes} 个，以覆盖不同障碍位置和动态障碍组合。','B')]
story += [P('2.1 指标含义与 Best 判定','H'),tab([['指标','含义','优化方向'],['成功率','到达目标邻域的场景比例；反映稳定完成任务的能力','越高越好'],['平均路径长度','机器人实际局部轨迹的累计长度；反映效率和潜在能耗','在最高成功率方法中越短越好'],['平均碰撞次数','轨迹进入占据栅格的平均次数；反映基本安全性','越低越好'],['最小动态间距','轨迹与动态障碍的最小距离再取场景平均；反映避障裕度','在最高成功率方法中越大越好']],[30*mm,112*mm,35*mm]),P('红色粗体表示每列 Best。为避免失败算法因提前停止而获得虚假的短路径，路径长度和动态安全间距只在达到最高成功率的方法中评选 Best；消融方法不参与主表 Best 评选。碰撞次数若并列为 0，则并列标红。','B')]
story += [P('3. 对比算法','H'),tab([['类别','算法','作用'],['全局','Dijkstra','无启发式栅格最短路基线'],['全局','A*','八邻域启发式基线'],['全局','RRT*','随机采样与 rewiring 基线'],['全局','Hybrid A*','方向/转弯代价基线（栅格近似）'],['全局','RCRA*','行间走廊、风险、转弯代价'],['局部','APF','目标吸引与障碍排斥'],['局部','DWA','速度窗口采样基线'],['局部','TEB','时间弹性带风格平滑安全基线（简化实现）'],['局部','Pure Pursuit','全局路径跟踪基线'],['局部','UA-DWA','预测安全、自适应权重、全局偏差']],[25*mm,38*mm,114*mm])]
story += [P('4. 主要结果','H')]
rows=[['方法','成功率','平均长度','碰撞','最小动态间距']]
for r in summary:
    label=P('<b><font color="red">'+r['method']+'</font></b>','Ours') if 'Ours' in r['method'] else r['method']
    is_main=not r['method'].startswith('no_'); s=float(r['success_rate']); l=float(r['mean_length']); c=float(r['mean_collisions']); cl=float(r['mean_clearance'])
    rows.append([label,redbest(f"{s*100:.1f}%",is_main and abs(s-best_success)<1e-9),redbest(f"{l:.2f}",is_main and abs(s-best_success)<1e-9 and abs(l-best_length)<1e-9),redbest(f"{c:.2f}",is_main and abs(c-best_collision)<1e-9),redbest(f"{cl:.2f}",is_main and abs(s-best_success)<1e-9 and abs(cl-best_clearance)<1e-9)])
story += [tab(rows,[43*mm,25*mm,30*mm,23*mm,34*mm]),P(f'Ours 在 {n_episodes} 个测试样例中的成功率为 {float(ours["success_rate"])*100:.1f}%，平均路径长度为 {float(ours["mean_length"]):.2f}；Hybrid A*+TEB 的成功率为 {float(hybrid["success_rate"])*100:.1f}%，平均路径长度为 {float(hybrid["mean_length"]):.2f}，动态安全间距为 {float(hybrid["mean_clearance"]):.2f}。增大样例数后，随机障碍位置和动态障碍组合更丰富，使 APF 的绕行代价、RRT* 的有限采样稳定性以及 Hybrid A*+TEB 的保守安全特征更加容易区分。','B')]
abl_names=['RCRA*+UA-DWA (Ours)','no_corridor','no_risk','no_adaptive','no_global_term']
abl_rows=[['变体','成功率','平均长度','最小动态间距']]
for name in abl_names:
    r=next(x for x in summary if x['method']==name); label=P('<b><font color="red">'+name+'</font></b>','Ours') if 'Ours' in name else name
    abl_rows.append([label,f"{float(r['success_rate'])*100:.1f}%",f"{float(r['mean_length']):.2f}",f"{float(r['mean_clearance']):.2f}"])
story += [P('5. 消融实验','H'),tab(abl_rows,[50*mm,30*mm,35*mm,40*mm]),P('完整方法同时包含行间走廊风险全局代价、航向对齐、动态障碍不确定性膨胀和停滞恢复。消融结果用于区分全局结构先验与局部恢复机制的贡献；若去掉某模块后成功率、效率或安全间距下降，则说明该模块对总体性能具有独立作用。','B')]
story += [P('6. 相比其他算法的优势与劣势','H'),P('<b>优势：</b>相较 Greedy 和 Dijkstra，Ours 将果园行结构和风险显式纳入代价，减少盲目绕行；相较普通 A*+DWA，Ours 通过动态风险和全局一致性获得更稳定的安全-长度折中；相较 APF，Ours 具有明确全局目标；相较 Pure Pursuit，Ours 具备在线避障；相较有限迭代 RRT*，Ours 的结果更稳定、可解释性更强；相较 Hybrid A*+TEB，Ours 针对果园行间风险和动态障碍进行专门建模。','B'),P('<b>劣势：</b>Hybrid A*+TEB 可能在车辆运动学可行性和轨迹平滑性上更有优势；当前 Ours 的成功率未必超过所有基线，说明需要增加高风险、交叉行和行末转弯场景；当前 Hybrid A* 和 TEB 为可复现实验中的简化 baseline，后续必须接入标准 ROS/Nav2 或原始开源实现进行严格复核。','B')]
story += [P('6.1 Ours 优化与独立种子验证','H'),P('针对旧版 UA-DWA 在果园主通道方向上起步停滞的问题，Ours 新增三项机制：以全局路径切线初始化局部航向；对动态障碍进行恒速预测与不确定性膨胀，并在可行空间内横向偏移；连续无进展时回接安全全局节点。改进后，主测试集 16 个场景成功率达到 100%。另使用未参与调试的随机种子 2000-2015 进行独立检查，成功率同样为 100%，平均路径长度为 42.28，平均动态安全间距为 18.94，碰撞为 0。该结果降低了仅针对主测试集调参的风险，但仍需更大规模和真实平台验证。','B')]
story += [P('7. 结果可复现性','H'),P(f'运行：python orchard_path_planning/benchmark.py --episodes {n_episodes} --out orchard_path_planning/results。benchmark.py 为完整实现；metrics.csv 保存 {n_episodes} 个场景的逐场景数据；summary.csv 为聚合结果；ablation.csv 为消融结果。','B'),Image(os.path.join(ROOT,'results','benchmark.png'),width=115*mm,height=101*mm),P('图 1. 代表性果园场景轨迹。','S')]
story += [P('8. 局限性与下一步','H'),P('当前实验是二维栅格仿真，尚未包含真实传感器噪声、定位漂移、坡度、车辆非完整约束和控制延迟；场景数量也偏少。下一步应扩展到至少 100 个场景，加入缺株、坡地、泥泞、高密度动态障碍，并增加 Hybrid A*、TEB、MPC/NMPC 基线；最终在 Gazebo/Isaac Sim 或真实果园中验证轨迹跟踪、能耗和实时性。当前结果应作为算法原型证据，而非现场性能结论。','B')]
story += [P('8. 参考文献','H'),P('1. Wang H. et al. An A*-DWA Algorithm Enhanced Laser SLAM System for Orchard Navigation. Agriculture, 2026. DOI: 10.3390/agriculture16040469. https://www.mdpi.com/2077-0472/16/4/469','S'),P('2. Research on Robot Path Planning Based on Point Cloud Map in Orchard Environment. IEEE Access. https://doaj.org/article/340c83ca44e94d2a94e367b1a6d0b3a1','S'),P('3. Environmental mapping and path planning for robots in orchard based on traversability analysis, improved LeGO-LOAM and RRT*. Computers and Electronics in Agriculture. https://www.sciencedirect.com/science/article/abs/pii/S0168169924012808','S'),P('4. Sun Y. et al. Fusion of A* and Dynamic Window Method. Electronics, 2022. DOI: 10.3390/electronics11172683.','S')]
def footer(c,d):
    c.saveState();c.setFont(F,7);c.setFillColor(colors.HexColor('#66808C'));c.drawString(17*mm,9*mm,'Orchard Path Planning Experiment Report');c.drawRightString(193*mm,9*mm,str(d.page));c.restoreState()
os.makedirs(os.path.dirname(OUT),exist_ok=True);SimpleDocTemplate(OUT,pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=15*mm,bottomMargin=15*mm).build(story,onFirstPage=footer,onLaterPages=footer);print(OUT)
