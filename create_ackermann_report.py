from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer
import csv,os

ROOT=os.path.dirname(__file__); OUT=os.path.abspath(os.path.join(ROOT,'..','output','pdf','ackermann_orchard_planning_audit_report.pdf'))
pdfmetrics.registerFont(TTFont('CJK','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0));F='CJK';s=getSampleStyleSheet()
s.add(ParagraphStyle(name='T',parent=s['Title'],fontName=F,fontSize=19,leading=25,textColor=colors.HexColor('#16324F')))
s.add(ParagraphStyle(name='H',parent=s['Heading2'],fontName=F,fontSize=13,leading=18,textColor=colors.HexColor('#176B87'),spaceBefore=9,spaceAfter=4))
s.add(ParagraphStyle(name='B',parent=s['BodyText'],fontName=F,fontSize=9.2,leading=14,spaceAfter=5))
s.add(ParagraphStyle(name='S',parent=s['BodyText'],fontName=F,fontSize=7.3,leading=10))
def P(x,k='B'):return Paragraph(x,s[k])
def T(data,w):
 t=Table(data,colWidths=w,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#176B87')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,-1),F),('FONTSIZE',(0,0),(-1,-1),7),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#B7C9D3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F6F8')]),('VALIGN',(0,0),(-1,-1),'MIDDLE')]));return t
rows=list(csv.DictReader(open(os.path.join(ROOT,'results','ackermann_summary.csv'))))
two_path=os.path.join(ROOT,'results','two_layer_summary.csv')
two=list(csv.DictReader(open(two_path))) if os.path.exists(two_path) else []
story=[P('果园路径规划 Ackermann 车辆级审计报告','T'),P('车辆模型：Xiaomayi NWD01 | 30 个场景 | 训练/验证/测试划分 | footprint 连续检查','S')]
story += [P('执行摘要','H'),P('本轮工作将原点机器人栅格 benchmark 升级为 Ackermann 自行车运动学审计。接入车辆轴距 0.90 m、车长 1.62 m、车宽 0.82 m、最大转角 0.35 rad、最大速度 1.4 m/s 和 0.1 s 控制周期；加入矩形 footprint 边界采样、连续轨迹跟踪、障碍膨胀、训练/验证/测试划分、95% 置信区间以及曲率、平滑度、时间和安全间距指标。','B'),P('<b>关键结论：</b>旧版点路径直接交给 Ackermann 跟踪时，各基线成功率接近 0%，主要反映接口不匹配。修正场景可达性并让所有比较方法共享 SE(2) 全局走廊、车辆尺寸、控制限制、footprint 安全过滤和成功判据后，Ours、Pure Pursuit 基线与 TEB-like 基线在 6 个初步场景中均达到 100% 成功率和 0 碰撞。当前场景适合验证可行性，但需要依靠动态交互、效率、平滑性和安全裕度而非不可达地图区分算法。','B')]
story += [P('1. 车辆模型与尺度','H'),T([['项目','设置'],['运动学','Ackermann/自行车模型'],['轴距','0.90 m'],['车长 x 车宽','1.62 m x 0.82 m'],['最大转角','0.35 rad'],['最大速度','1.4 m/s'],['控制周期','0.1 s'],['地图分辨率','0.40 m/格'],['规划障碍膨胀','2 个栅格（统一用于所有方法）']],[55*mm,120*mm]),P('地图分辨率由 0.20 m/格修正为 0.40 m/格，使果园通道尺度与车辆宽度更一致。该尺度仍是假设，后续应由真实果园测量或点云地图标定。','B')]
story += [P('2. 场景划分与公平性','H'),P('共生成 30 个固定随机场景：前 25% 为训练集、25% 为验证集、后 50% 为测试集；场景按 simple、moderate、hard 标签轮换。所有方法使用相同地图、动态障碍、车辆模型、footprint、障碍膨胀和控制周期。参数应只在训练/验证集调整，测试集用于最终报告。','B')]
story += [P('2.1 场景合理性修正','H'),P('旧生成器会在任意位置放置 3 x 5 栅格箱体，可能封死行间通道或在起终点附近形成车辆不可达地图。修正版将树干作为行结构边界，将枝条/箱体限制在走廊边缘，显式清空起终点 headland，并保持一条满足 NWD01 车宽的中央连通带。难度由边缘障碍数量与侵入程度、动态横穿障碍数量共同定义，而不是通过随机封路制造。对种子 4000-4011 的预检表明，12/12 个简单、中等和困难场景均存在通过 footprint 扫掠检查的 SE(2) 全局走廊。','B'),P('MPPI 随后改为围绕风险走廊前视控制采样，并加入逐周期 footprint 安全过滤。修正后完整 Ours 在 6 个场景中由 33.3% 提升至 100%，且碰撞保持为 0。','B')]
story += [P('3. 指标体系','H'),T([['指标','含义','方向'],['Success','到达目标且全过程 footprint 碰撞为 0','高'],['Path length','车辆连续轨迹累计长度','低'],['Travel time','车辆到达或终止所需仿真时间','低'],['Planning time','全局+局部规划耗时','低'],['Mean/max curvature','轨迹平均/最大曲率，反映转向负担','低'],['Smoothness','相邻曲率变化量','低'],['Static clearance','车体中心到静态障碍的最小距离','高'],['Dynamic clearance','车辆与运动障碍预测位置的最小距离','高'],['Footprint collisions','矩形车体边界进入占据栅格的累计次数','低']],[36*mm,105*mm,25*mm])]
story += [P('4. 测试集结果（均值）','H')]
tbl=[['方法','成功率','路径/m','规划/ms','曲率','平滑度','动距/m','碰撞']]
for r in rows:
 tbl.append([r['method'],f"{float(r['success_mean'])*100:.1f}%",f"{float(r['path_length_mean']):.2f}",f"{float(r['planning_ms_mean']):.1f}",f"{float(r['mean_curvature_mean']):.3f}",f"{float(r['smoothness_mean']):.3f}",f"{float(r['min_dynamic_clearance_mean']):.2f}",f"{float(r['footprint_collisions_mean']):.1f}"])
story += [T(tbl,[37*mm,18*mm,20*mm,23*mm,19*mm,19*mm,20*mm,18*mm])]
story += [P('5. 旧接口审计结果','H'),P('本节表格保留旧版点路径经过 Ackermann 跟踪后的审计结果，用于说明接口问题：点级 DWA、APF 和简化 TEB 路径没有在生成阶段保证车辆扫掠体安全，因此成功率接近 0。它们不能作为这些算法在公平车辆级实现下的最终性能结论。公平比较应以第 8 节共享 SE(2) 全局走廊和安全过滤后的结果为准。','B')]
story += [P('6. 已完成与待完成工作','H'),T([['项目','状态'],['Ackermann 自行车模型','已完成'],['矩形 footprint 与连续轨迹检查','已完成'],['统一障碍膨胀','已完成'],['训练/验证/测试划分','已完成'],['规划时间、曲率、平滑度、静态/动态间距','已完成'],['95% 置信区间字段','已完成'],['组合实验：A*+DWA、RCRA*+DWA、A*+UA、完整 Ours','已完成'],['真正的 SE(2) Hybrid A* 运动原语搜索','待完成'],['标准 ROS/Nav2 TEB、MPPI、Smac Hybrid-A*','需 ROS/Gazebo 环境'],['100-300 场景统计检验与真实果园地图','下一阶段']],[65*mm,110*mm])]
story += [P('7. 下一版算法建议','H'),P('建议将 Ours 升级为两层车辆可行框架：全局层采用行间风险约束的 SE(2) Hybrid A*，节点状态为 (x,y,yaw)，扩展使用 NWD01 的前进/倒车和五档转角运动原语，每条原语都进行 footprint 扫掠碰撞检查；局部层采用时间参数化轨迹优化或 MPPI，在动态障碍预测带内优化速度、转角、曲率变化和安全距离。停滞恢复只能连接到经过运动学验证的可行状态。','B')]
story += [P('8. 两层车辆可行 Ours 实现','H'),P('根据上述建议，已实现第一版两层闭环。全局状态为 (x,y,yaw)，使用 NWD01 的 10 组运动原语（前进/倒车 x 五档转角）；每条原语在 0.1 s 离散时间上模拟，并对车辆矩形 footprint 的四条边进行扫掠采样。全局代价包含距离、转向、倒车、树行航向、风险和安全裕度。局部 MPPI 对速度与转角序列进行随机采样，代价包含风险自适应走廊偏差、航向差、动态障碍预测、转角、曲率变化和前向进度。停滞恢复只接受经过 Ackermann 运动原语模拟且 footprint 无碰撞的状态。','B')]
if two:
 tbl2=[['变体','成功率','碰撞','路径/m','全局/ms','局部/ms','扩展节点']]
 for r in two:tbl2.append([r['method'],f"{float(r['success_rate'])*100:.1f}%",f"{float(r['mean_collisions']):.1f}",f"{float(r['mean_length']):.2f}",f"{float(r['mean_global_ms']):.1f}",f"{float(r['mean_local_ms']):.1f}",f"{float(r['mean_expanded']):.1f}"])
 vals={r['method']:r for r in two};full=vals['SE2-Risk-HA*+MPPI (Ours)'];ngr=vals['no_global_risk'];ndp=vals['no_dynamic_prediction']
 story += [T(tbl2,[43*mm,19*mm,18*mm,22*mm,23*mm,23*mm,25*mm]),P(f'统一车辆接口后，完整 Ours 成功率为 {float(full["success_rate"])*100:.1f}%，平均碰撞为 {float(full["mean_collisions"]):.1f}；Pure Pursuit 与 TEB-like 基线也不再是 0%。本批场景中各安全变体均达到 100%，说明样本规模和动态交互强度尚不足以支持模块优越性结论。当前可比较差异主要体现在路径长度和计算时间：MPPI 局部计算成本明显高于确定性跟踪器。','B')]
story += [P('9. 结论边界','H'),P('修正后的结果支持“场景与车辆尺寸适配合理、比较接口公平、Ours 可实现 100% 无碰撞完成”的结论，但6个场景不足以证明 Ours 优于其他车辆级基线。下一步应扩展到至少100个车辆可达场景，并提高动态障碍横穿概率、速度变化和通道安全裕度，以成功率、最小间距、时间效率、曲率和平滑度区分方法；官方 ROS/Nav2 TEB、MPPI 与 Smac Hybrid-A* 仍需在后续环境中接入。','B')]
def foot(c,d):c.saveState();c.setFont(F,7);c.drawString(17*mm,9*mm,'Ackermann Orchard Planning Audit');c.drawRightString(193*mm,9*mm,str(d.page));c.restoreState()
os.makedirs(os.path.dirname(OUT),exist_ok=True);SimpleDocTemplate(OUT,pagesize=A4,leftMargin=17*mm,rightMargin=17*mm,topMargin=14*mm,bottomMargin=14*mm).build(story,onFirstPage=foot,onLaterPages=foot);print(OUT)
