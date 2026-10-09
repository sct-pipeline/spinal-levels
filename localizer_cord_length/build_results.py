#!/usr/bin/env python3
"""Rebuild all cord-length deliverables from the current labels:
  - merged CSV (cord length + L1-S2 & L1-L5 spans + height/weight/sex/BMI)
  - correlations
  - 3-panel matplotlib figure (cord_length_comparison.png)
  - update the Excel: cord-length column + dedicated sheet + by-sex scatter w/ trendline
Run after any conus/length change. Uses /usr/bin/python3 (pandas+openpyxl+scipy+matplotlib).
"""
import csv, statistics as st, os
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import openpyxl
from openpyxl.chart import ScatterChart, Series, Reference
from openpyxl.chart.trendline import Trendline
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter

BIDS=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
XL="/Users/eliarochiccioli/PhD/Lumbar registration /spinal_level_heights_top_L1-bottom_S2 .xlsx"
LAB=os.path.join(BIDS,"derivatives/labels")

L={r["participant_id"]:float(r["cord_length_mm"]) for r in csv.DictReader(open(f"{LAB}/cord_length.csv"))}
wb=openpyxl.load_workbook(XL); ws=wb["Spinal level heights (mm)"]; dem=wb["Demographics"]
demo={}
for r in range(2,dem.max_row+1):
    pid=dem.cell(r,1).value
    if not pid: continue
    w=dem.cell(r,4).value; h=dem.cell(r,5).value
    demo[pid]=dict(sex=(dem.cell(r,2).value or "").strip().lower(),
                   weight=w if isinstance(w,(int,float)) else None, height=h if isinstance(h,(int,float)) else None)
span_s2={}; span_l5={}
for r in range(2,19):
    pid=ws.cell(r,1).value
    if pid and str(pid).startswith("sub-ltr"):
        v=[ws.cell(r,c).value for c in range(2,9)]
        if all(isinstance(x,(int,float)) for x in v): span_s2[pid]=round(sum(v),1); span_l5[pid]=round(sum(v[:5]),1)

order=[f"sub-ltr{n:02d}" for n in [1,2,3,4,5,6,7,8,9,11,13,14,15,16,17,19,20,21,23,24,25]]
rows=[]
for pid in order:
    if pid not in L: continue
    d=demo.get(pid,{}); h=d.get("height"); w=d.get("weight")
    rows.append(dict(pid=pid,cord=round(L[pid],1),s2=span_s2.get(pid,""),l5=span_l5.get(pid,""),
                     height=h or "",weight=w or "",sex=d.get("sex",""),
                     bmi=round(w/((h/100)**2),1) if (h and w) else ""))
with open(f"{LAB}/cord_length_vs_spinal_levels.csv","w",newline="") as f:
    wri=csv.writer(f);wri.writerow(["participant_id","cord_length_mm","L1_S2_span_mm","L1_L5_span_mm","body_height_cm","weight_kg","sex","bmi"])
    for r in rows: wri.writerow([r["pid"],r["cord"],r["s2"],r["l5"],r["height"],r["weight"],r["sex"],r["bmi"]])

def corr(xk,yk):
    xy=[(r[xk],r[yk]) for r in rows if isinstance(r[xk],(int,float)) and isinstance(r[yk],(int,float))]
    if len(xy)<3: return None
    x,y=zip(*xy); r_,p=stats.pearsonr(x,y); return (r_,p,len(xy),np.array(x),np.array(y))
c_h=corr("height","cord"); c_s2=corr("s2","cord"); c_l5=corr("l5","cord")
for lab,c in [("cord vs height",c_h),("cord vs L1-S2",c_s2),("cord vs L1-L5",c_l5)]:
    print(f"{lab}: r={c[0]:+.2f} p={c[1]:.3f} n={c[2]}")

# --- matplotlib 3-panel figure ---
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for ax,(c,xl,yl,ttl) in zip(axes,[(c_h,"Body height (cm)","Cord length PMJ→conus (mm)","vs body height"),
                                   (c_s2,"L1–S2 span (mm)","Cord length PMJ→conus (mm)","vs L1–S2 span"),
                                   (c_l5,"L1–L5 span (mm)","Cord length PMJ→conus (mm)","vs L1–L5 span")]):
    x,y=c[3],c[4];ax.scatter(x,y,s=42,color="#2c7fb8",edgecolor="white",zorder=3)
    b1,b0=np.polyfit(x,y,1);xs=np.linspace(x.min(),x.max(),50);ax.plot(xs,b0+b1*xs,color="#d95f02",lw=1.8)
    ax.set_xlabel(xl);ax.set_ylabel(yl);ax.grid(alpha=.25)
    ax.set_title(f"{ttl}\nr={c[0]:+.2f}{'*' if c[1]<.05 else ''} (p={c[1]:.3f}), n={c[2]}",fontsize=9)
plt.suptitle(f"LTR cord length (PMJ→conus) — n={len(rows)} participants",fontsize=11,y=1.03)
plt.tight_layout();plt.savefig(f"{LAB}/cord_length_comparison.png",dpi=140,bbox_inches="tight")

# --- rebuild Excel sheet + chart ---
if "Cord length (PMJ-conus)" in wb.sheetnames: del wb["Cord length (PMJ-conus)"]
ns=wb.create_sheet("Cord length (PMJ-conus)")
HF=Font(bold=True,sz=11,color="FFFFFFFF");HFill=PatternFill("solid",fgColor="FF1F4E78")
thin=Side(style="thin");B=Border(left=thin,right=thin,top=thin,bottom=thin);CEN=Alignment(horizontal="center");LEFT=Alignment(horizontal="left")
titles=["Participant","Cord length PMJ-conus (mm)","L1-S2 span (mm)","L1-L5 span (mm)","Body height (cm)","Weight (kg)","Sex","BMI"]
for j,t in enumerate(titles,1):
    c=ns.cell(1,j,t);c.font=HF;c.fill=HFill;c.alignment=CEN;c.border=B
for i,r in enumerate(rows,2):
    ns.cell(i,1,r["pid"]).border=B
    for j,k in zip(range(2,9),["cord","s2","l5","height","weight","sex","bmi"]):
        cell=ns.cell(i,j,r[k] if r[k]!="" else None);cell.border=B;cell.alignment=LEFT if j==7 else CEN
        if j in(2,3,4,5,6,8) and isinstance(r[k],(int,float)): cell.number_format="0.0"
nr=len(rows)+1
allc=[r["cord"] for r in rows];fc=[r["cord"] for r in rows if r["sex"]=="f"];mc=[r["cord"] for r in rows if r["sex"]=="m"]
summ=[("Summary",""),("n",len(allc)),("Mean (mm)",round(st.mean(allc),1)),("SD (mm)",round(st.pstdev(allc),1)),
      ("Min (mm)",round(min(allc),1)),("Max (mm)",round(max(allc),1)),
      ("Female mean±SD",f"{st.mean(fc):.1f} ± {st.pstdev(fc):.1f} (n={len(fc)})"),
      ("Male mean±SD",f"{st.mean(mc):.1f} ± {st.pstdev(mc):.1f} (n={len(mc)})"),("",""),
      ("Correlations (Pearson r, p)",""),
      ("cord length vs body height",f"r={c_h[0]:+.2f}, p={c_h[1]:.3f} (n={c_h[2]})"),
      ("cord length vs L1-S2 span",f"r={c_s2[0]:+.2f}, p={c_s2[1]:.3f} (n={c_s2[2]})"),
      ("cord length vs L1-L5 span",f"r={c_l5[0]:+.2f}, p={c_l5[1]:.3f} (n={c_l5[2]})")]
for di,(a,b) in enumerate(summ):
    rr=nr+2+di;ns.cell(rr,1,a).font=Font(bold=True,italic=a.startswith(("Summary","Correlations")));ns.cell(rr,2,b)
for j,w in enumerate([16,26,16,16,16,12,8,10],1): ns.column_dimensions[get_column_letter(j)].width=w
for c,t in {17:"H_female",18:"Cord_female",19:"H_male",20:"Cord_male"}.items():
    ns.cell(1,c,t).font=Font(italic=True,sz=9,color="FF808080");ns.column_dimensions[get_column_letter(c)].width=11
for i,r in enumerate(rows,2):
    if isinstance(r["height"],(int,float)):
        if r["sex"]=="f": ns.cell(i,17,r["height"]);ns.cell(i,18,r["cord"])
        elif r["sex"]=="m": ns.cell(i,19,r["height"]);ns.cell(i,20,r["cord"])
first,last=2,len(rows)+1
ch=ScatterChart();ch.title="Cord length (PMJ→conus) vs body height — by sex"
ch.x_axis.title="Body height (cm)";ch.y_axis.title="Cord length PMJ→conus (mm)"
ch.x_axis.delete=False;ch.y_axis.delete=False
ch.x_axis.scaling.min=155;ch.x_axis.scaling.max=190;ch.y_axis.scaling.min=425;ch.y_axis.scaling.max=515
ch.height=9.5;ch.width=16
def pts(xc,yc,label,color):
    s=Series(Reference(ns,min_col=yc,min_row=first,max_row=last),Reference(ns,min_col=xc,min_row=first,max_row=last),title=label)
    m=Marker(symbol="circle",size=7);m.graphicalProperties=GraphicalProperties(solidFill=color);s.marker=m;s.graphicalProperties.line.noFill=True;return s
ch.series.append(pts(17,18,"Female","E377C2"));ch.series.append(pts(19,20,"Male","1F77B4"))
s_all=Series(Reference(ns,min_col=2,min_row=first,max_row=last),Reference(ns,min_col=5,min_row=first,max_row=last),title="Trend (all)")
s_all.marker=Marker(symbol="none");s_all.graphicalProperties.line.noFill=True
tl=Trendline(trendlineType="linear",dispRSqr=True,dispEq=True);tl.graphicalProperties=GraphicalProperties();tl.graphicalProperties.line=LineProperties(solidFill="000000",w=19050);s_all.trendline=tl
ch.series.append(s_all);ch.legend.position="r";ns.add_chart(ch,"J2")
wb.save(XL)
print(f"\nmean {st.mean(allc):.1f} ± {st.pstdev(allc):.1f} mm (n={len(rows)}); Excel + figure + CSV rebuilt")
