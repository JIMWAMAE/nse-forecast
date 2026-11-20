import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(16, 7.2)); ax.set_xlim(0, 16); ax.set_ylim(0, 7.2); ax.axis("off")
TEAL, DARK, GREEN, GOLD, RED, LIGHT, INK, MUTE = "#1C6E6A","#0F2A2E","#2BB673","#E9A23B","#E4572E","#EEF4F2","#1F2933","#5B6770"
def box(x,y,w,h,title,body,fc=LIGHT,tc=TEAL,bc=INK,ec="none"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.15",fc=fc,ec=ec,lw=1.5))
    ax.text(x+w/2,y+h-0.35,title,ha="center",va="top",fontsize=13,weight="bold",color=tc)
    ax.text(x+w/2,y+h-0.85,body,ha="center",va="top",fontsize=10.5,color=bc,linespacing=1.5)
def arrow(x1,y1,x2,y2,c=MUTE):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=22,lw=2.2,color=c))
def step(x,n,label,c):
    ax.add_patch(plt.Circle((x,6.55),0.27,color=c)); ax.text(x,6.55,str(n),ha="center",va="center",color="white",weight="bold",fontsize=13)
    ax.text(x+0.4,6.55,label,va="center",fontsize=14,weight="bold",color=DARK)
ax.text(8,7.25,"How the NSE forecasting system works",ha="center",fontsize=18,weight="bold",color=DARK)
step(0.55,1,"Data",TEAL); step(3.75,2,"Train (make pipeline)",TEAL); step(10.15,3,"Serve (Docker)",GREEN); step(13.6,4,"Use",GOLD)
box(0.3,2.3,2.6,3.7,"NSE Excel files","data/raw/\n\n10 stocks\nMay 2015 – Jun 2026\n\nprice, high, low,\nvolume …")
# pipeline
y=4.45
box(3.5,y,2.7,1.55,"1. Clean","remove duplicates\nand jumps > 15%")
box(6.5,y,2.7,1.55,"2. Features","returns, momentum,\nvolatility, RSI, volume")
box(3.5,2.3,2.7,1.75,"3. EGARCH","one per stock\npredicts tomorrow's\nvolatility (risk)")
box(6.5,2.3,2.7,1.75,"4. Hybrid XGBoost","features + EGARCH\npredicts tomorrow's\nreturn", fc=DARK, tc=GREEN, bc="white")
arrow(2.95,4.15,3.45,4.15); arrow(6.25,5.22,6.45,5.22); arrow(7.85,4.42,7.85,4.1); arrow(6.25,3.17,6.45,3.17); arrow(4.85,4.42,4.85,4.1)
ax.add_patch(FancyBboxPatch((3.35,2.15),6.0,4.0,boxstyle="round,pad=0.02,rounding_size=0.2",fc="none",ec=TEAL,lw=1.6,ls="--"))
# model store
box(5.0,0.25,2.7,1.45,"Saved models","trained model files\nused by the API",fc="white",ec=TEAL)
arrow(7.85,2.25,7.0,1.75)
# docker
ax.add_patch(FancyBboxPatch((9.75,0.9),3.3,5.25,boxstyle="round,pad=0.02,rounding_size=0.2",fc="#E6F6EE",ec=GREEN,lw=2))
ax.text(11.4,5.9,"Docker containers",ha="center",fontsize=12,weight="bold",color=GREEN)
box(10.0,3.55,2.8,2.05,"FastAPI","port 8000\n/forecast → JSON\nfor stock platforms",fc="white")
box(10.0,1.15,2.8,2.05,"Streamlit app","port 8501\nruns the same model steps\nfor the chosen stock\n→ chart + 80% range",fc="white")
arrow(7.75,1.2,9.95,4.4); arrow(9.25,2.6,9.95,2.2)
# users
box(13.5,3.55,2.2,2.05,"Stock platform","calls the API",fc="#FDF3E1",tc="#B7791F")
box(13.5,1.15,2.2,2.05,"Investor","opens the app\nin a browser",fc="#FDF3E1",tc="#B7791F")
arrow(12.85,4.57,13.45,4.57); arrow(12.85,2.17,13.45,2.17)
ax.text(8,0.05,"Every forecast uses only data up to the day it is made. Train: to 2023 · Validate: 2024–2025 · Test: Jan–Jun 2026",ha="center",fontsize=10.5,style="italic",color=MUTE)
plt.savefig("docs/images/architecture.png",dpi=150,bbox_inches="tight",facecolor="white")
