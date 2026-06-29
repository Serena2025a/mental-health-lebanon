import streamlit as st

# --- Password Protection ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("""
            <div style='text-align:center; padding: 80px 0 20px 0;'>
                <h1 style='color:#2d6a4f; font-size:2rem; font-weight:700;'>
                    Mental Health Burden in Post-Crisis Lebanon
                </h1>
                <p style='color:#555; font-size:1rem; margin-top:8px;'>
                    MSBA382 – Healthcare Analytics | AUB Suliman S. Olayan School of Business
                </p>
                <p style='color:#555; font-size:0.9rem;'>
                    Enter the password to access the dashboard.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            pwd = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter password")
            if st.button("Login", use_container_width=True):
                if pwd == "msba382":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        return False
    return True

if not check_password():
    st.stop()

# --- Rest of your app below ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt
 
st.set_page_config(page_title="Mental Health Lebanon", page_icon=":brain:",
                   layout="wide", initial_sidebar_state="expanded")
 
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background-color:#f8faf8;}
h1,h2,h3 {color:#1a1a1a!important;}
.sec {
    background:linear-gradient(90deg,#2d6a4f,#52b788);
    padding:3px 10px; border-radius:5px;
    margin:4px 0 3px 0;
    color:white!important; font-weight:bold; font-size:0.8rem; line-height:1.6;
}
.kcard {
    background:#ffffff; border-radius:8px;
    padding:7px 6px; text-align:center;
    border-top:3px solid;
    box-shadow: 0 1px 5px rgba(0,0,0,0.07);
}
.kval {font-size:1.15rem; font-weight:bold; color:#1a1a1a;}
.klbl {font-size:0.6rem; color:#555; margin-top:2px; line-height:1.2;}
.kdlt {font-size:0.62rem; margin-top:2px;}
section[data-testid="stSidebar"] {background-color:#f0f7f4;}
</style>
""", unsafe_allow_html=True)
 
@st.cache_data
def load():
    gbd = pd.read_csv("IHME-GBD_2023_DATA-2c566f3e-1.csv")
    wb  = pd.read_csv("World Bank CSV file.csv", encoding="latin1")
    return gbd, wb
 
df, wb = load()
 
ISO  = {"Lebanon":"LBN","Egypt":"EGY","Iraq":"IRQ","Jordan":"JOR",
        "Saudi Arabia":"SAU","Syrian Arab Republic":"SYR","Iran (Islamic Republic of)":"IRN"}
DISP = {"Syrian Arab Republic":"Syria","Iran (Islamic Republic of)":"Iran",
        "Saudi Arabia":"Saudi Arabia","Lebanon":"Lebanon",
        "Egypt":"Egypt","Iraq":"Iraq","Jordan":"Jordan"}
MEAS = {"Prevalence":"Prevalence",
        "DALYs":"DALYs (Disability-Adjusted Life Years)",
        "YLDs":"YLDs (Years Lived with Disability)",
        "YLLs":"YLLs (Years of Life Lost)"}
CAUS = {"All Mental Disorders":"Mental disorders",
        "Depressive Disorders":"Depressive disorders",
        "Anxiety Disorders":"Anxiety disorders",
        "Self-harm":"Self-harm"}
AGE_ORD = ["5-14 years","15-19 years","20-54 years","55+ years"]
AGE_LBL = ["Children (5-14)","Youth (15-19)","Working Age (20-54)","Older Adults (55+)"]
YRS_MH  = list(range(1990,2024))
 
def val(cause, measure, sex, year, loc="Lebanon"):
    r = df[(df.location==loc)&(df.cause==cause)&(df.measure==measure)&
           (df.sex==sex)&(df.age=="All ages")&(df.metric=="Rate")&(df.year==year)]
    return float(r["val"].values[0]) if len(r) else None
 
def parse_wb(wb_df, series_name):
    row = wb_df[wb_df["Series Name"]==series_name].iloc[0]
    yr_cols = [c for c in wb_df.columns if "[YR" in c]
    years, vals = [], []
    for col in yr_cols:
        yr = int(col.split("[YR")[1].replace("]",""))
        v  = row[col]
        try:
            fv = float(v)
            if not pd.isna(fv) and 2009 <= yr <= 2023:
                years.append(yr); vals.append(fv)
        except: pass
    return pd.DataFrame({"year":years,"value":vals}).sort_values("year")
 
# ── SIDEBAR
with st.sidebar:
    st.markdown("### Dashboard Filters")
    st.markdown("---")
    s_yr  = st.slider("Year", 1990, 2023, 2023)
    s_sx  = st.selectbox("Gender", ["Both","Male","Female"])
    s_msl = st.selectbox("Measure", list(MEAS.keys()))
    s_ms  = MEAS[s_msl]
    s_dcl = st.selectbox("Disorder", list(CAUS.keys()))
    s_dc  = CAUS[s_dcl]
    st.markdown("---")
    st.caption("MSBA382 Healthcare Analytics - Serena Aoun - Data: IHME GBD 2023 + World Bank")
 
# ── TITLE
st.markdown("""
<h1 style='text-align:center;color:#1a1a1a;font-size:1.3rem;margin-bottom:0;padding:0'>
Mental Health Burden in Post-Crisis Lebanon
</h1>
<p style='text-align:center;color:#555;font-size:0.72rem;margin-top:1px;margin-bottom:3px'>
IHME GBD 2023 - World Bank - MSBA382 Healthcare Analytics
</p>
""", unsafe_allow_html=True)
 
# ── ROW 1: KPI CARDS
st.markdown("<div class='sec'>Key Indicators | Lebanon</div>", unsafe_allow_html=True)
KPIS = [("Depressive Disorders","Depressive disorders","#2d6a4f"),
        ("Anxiety Disorders","Anxiety disorders","#1d8ed8"),
        ("All Mental Disorders","Mental disorders","#52b788"),
        ("Self-harm","Self-harm","#f7b731")]
k1,k2,k3,k4 = st.columns(4)
for col,(lbl,cause,color) in zip([k1,k2,k3,k4], KPIS):
    v  = val(cause, s_ms, s_sx, s_yr)
    vp = val(cause, s_ms, s_sx, s_yr-1)
    if v:
        if vp and vp>0:
            d  = ((v-vp)/vp)*100
            ds = f"+{d:.1f}% vs {s_yr-1}" if d>0 else f"{d:.1f}% vs {s_yr-1}"
            dc = "#c62828" if d>0 else "#2d6a4f"
        else:
            ds,dc = "","#aaa"
        with col:
            st.markdown(f"""
            <div class='kcard' style='border-top-color:{color}'>
                <div class='kval'>{v:,.0f}</div>
                <div class='klbl'>per 100k - {s_msl}<br>{lbl}</div>
                <div class='kdlt' style='color:{dc}'>{ds}</div>
            </div>""", unsafe_allow_html=True)
 
# ── ROW 2: MAP | AGE GROUP | GENDER
r2a, r2b, r2c = st.columns([1.8, 1.5, 1.1])
 
# MAP
with r2a:
    st.markdown("<div class='sec'>MENA Region | Mental Health Burden Map</div>", unsafe_allow_html=True)
    mdf = df[(df.year==s_yr)&(df.metric=="Rate")&(df.sex==s_sx)&
             (df.age=="All ages")&(df.measure==s_ms)&
             (df.cause==s_dc)&(df.location!="Global")].copy()
    mdf["iso"]  = mdf["location"].map(ISO)
    mdf["name"] = mdf["location"].map(DISP).fillna(mdf["location"])
    mdf = mdf.dropna(subset=["iso"])
    mdf["Male"]   = mdf["location"].apply(lambda l: f"{val(s_dc,s_ms,'Male',s_yr,l):,.0f}" if val(s_dc,s_ms,"Male",s_yr,l) else "N/A")
    mdf["Female"] = mdf["location"].apply(lambda l: f"{val(s_dc,s_ms,'Female',s_yr,l):,.0f}" if val(s_dc,s_ms,"Female",s_yr,l) else "N/A")
    fig_map = px.choropleth(mdf, locations="iso", color="val",
        hover_name="name", hover_data={"iso":False,"val":":.0f","Male":True,"Female":True},
        color_continuous_scale=["#d8f3dc","#95d5b2","#52b788","#2d6a4f"],
        labels={"val":"Rate per 100k"}, title=f"{s_dcl} - {s_msl} ({s_yr})")
    fig_map.update_geos(scope="asia", center={"lat":29,"lon":44}, projection_scale=3.8,
        showland=True, landcolor="#e8f4e8", showocean=True, oceancolor="#dce8f5",
        showcountries=True, countrycolor="#555", bgcolor="#f0f8f0")
    fig_map.update_layout(paper_bgcolor="#ffffff", font_color="#1a1a1a", height=260,
        margin=dict(l=0,r=0,t=35,b=0),
        coloraxis_colorbar=dict(tickfont=dict(color="#1a1a1a"),
                                title=dict(text="Rate/100k", font=dict(color="#1a1a1a"))))
    st.plotly_chart(fig_map, width="stretch")
 
# AGE GROUP
with r2b:
    st.markdown("<div class='sec'>Burden by Age Group | Lebanon</div>", unsafe_allow_html=True)
    ad = df[(df.location=="Lebanon")&(df.cause==s_dc)&(df.measure==s_ms)&
            (df.sex==s_sx)&(df.metric=="Rate")&(df.year==s_yr)&
            (df.age.isin(AGE_ORD))].copy()
    ad["ord"] = ad["age"].map({a:i for i,a in enumerate(AGE_ORD)})
    ad = ad.sort_values("ord")
    ad["lbl"] = ad["age"].map(dict(zip(AGE_ORD, AGE_LBL)))
    fig_age = go.Figure(go.Bar(
        x=ad["lbl"], y=ad["val"],
        marker_color=["#52b788","#2d6a4f","#1d8ed8","#7b2d8b"],
        text=ad["val"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside", textfont=dict(color="#1a1a1a", size=9),
        hovertemplate="%{x}: %{y:,.0f} per 100k<extra></extra>"
    ))
    fig_age.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font_color="#1a1a1a", height=260,
        title=dict(text=f"{s_dcl} - {s_msl} ({s_yr})", font=dict(size=11, color="#1a1a1a"), x=0),
        margin=dict(l=40,r=10,t=35,b=45),
        xaxis=dict(showgrid=False, tickfont=dict(size=8)),
        yaxis=dict(showgrid=False, showline=True, linecolor="#cccccc",
                   title="Rate per 100,000", tickfont=dict(size=8)))
    st.plotly_chart(fig_age, width="stretch")
 
# GENDER
with r2c:
    st.markdown("<div class='sec'>Gender Split | Lebanon</div>", unsafe_allow_html=True)
    male_v   = val(s_dc, s_ms, "Male",   s_yr)
    female_v = val(s_dc, s_ms, "Female", s_yr)
    if male_v is not None and female_v is not None:
        total = male_v + female_v
        fig_pie = go.Figure(go.Pie(
            labels=["Male","Female"], values=[male_v, female_v], hole=0.55,
            marker=dict(colors=["#1d8ed8","#7b2d8b"], line=dict(color="#ffffff",width=2)),
            textinfo="label+percent", textfont=dict(size=10, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} per 100k<br>%{percent}<extra></extra>"
        ))
        fig_pie.update_layout(paper_bgcolor="#ffffff", font_color="#1a1a1a",
            height=260, showlegend=False,
            annotations=[dict(text=f"{total:,.0f}<br>per 100k<br><span style='font-size:9px;color:#888'>{s_msl} | {s_dcl}</span>",
                              x=0.5, y=0.5, font=dict(size=11, color="#2d6a4f"), showarrow=False)],
            margin=dict(t=10,b=10,l=5,r=5))
        st.plotly_chart(fig_pie, width="stretch")
    else:
        st.warning("No data.")
 
# ── ROW 3: ECONOMIC TREND | FORECAST
r3a, r3b = st.columns([1, 1])
 
# ECONOMIC TREND
with r3a:
    st.markdown("<div class='sec'>Mental Health Prevalence & Economic Indicators (2009-2023) | Lebanon</div>", unsafe_allow_html=True)
    ECO = {
        "Inflation (%)":{"series":"Inflation, consumer prices (annual %)","color":"#1d8ed8","hover":"%{y:.1f}%"},
        "Unemployment (%)":{"series":"Unemployment, total (% of total labor force) (modeled ILO estimate)","color":"#e07b00","hover":"%{y:.1f}%"},
        "Health Expenditure per Capita (USD)":{"series":"Current health expenditure per capita (current US$)","color":"#7b2d8b","hover":"$%{y:,.0f}"}
    }
    mh_tr = df[(df.location=="Lebanon")&(df.cause=="Mental disorders")&
               (df.measure=="Prevalence")&(df.sex=="Both")&
               (df.age=="All ages")&(df.metric=="Rate")&
               (df.year>=2009)&(df.year<=2023)].sort_values("year")
    mh_tr_glb = df[(df.location=="Global")&(df.cause=="Mental disorders")&
                   (df.measure=="Prevalence")&(df.sex=="Both")&
                   (df.age=="All ages")&(df.metric=="Rate")&
                   (df.year>=2009)&(df.year<=2023)].sort_values("year")
    eco_pick = st.selectbox("Indicator to overlay:", list(ECO.keys()), key="eco_pick")
    eco_info = ECO[eco_pick]
    eco_df   = parse_wb(wb, eco_info["series"])
    _merged  = pd.merge(mh_tr[["year","val"]], eco_df, on="year", how="inner")
    if len(_merged) >= 3:
        _r = _merged["val"].corr(_merged["value"])
        _s = "very strong" if abs(_r)>=0.8 else "strong" if abs(_r)>=0.6 else "moderate" if abs(_r)>=0.4 else "weak"
        _d = "positive" if _r>0 else "negative"
        st.caption(f"Pearson r = {_r:+.2f} ({_s} {_d}). Correlation reflects co-movement, not causation.")
    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=mh_tr["year"], y=mh_tr["val"], name="Mental Health (Lebanon)",
        line=dict(color="#2d6a4f",width=2.5), mode="lines+markers", marker=dict(size=5,color="#2d6a4f"),
        yaxis="y1", hovertemplate="<b>%{x}</b>: %{y:,.0f} per 100k<extra>Lebanon</extra>"))
    fig_tr.add_trace(go.Scatter(x=mh_tr_glb["year"], y=mh_tr_glb["val"], name="Global Rate",
        line=dict(color="#888888",width=1.5,dash="dash"), mode="lines", yaxis="y1",
        hovertemplate="<b>%{x}</b>: %{y:,.0f} per 100k<extra>Global</extra>"))
    fig_tr.add_trace(go.Scatter(x=eco_df["year"], y=eco_df["value"], name=eco_pick,
        line=dict(color=eco_info["color"],width=2.5), mode="lines+markers",
        marker=dict(size=5,color=eco_info["color"]), yaxis="y2",
        hovertemplate=f"<b>%{{x}}</b>: {eco_info['hover']}<extra>{eco_pick}</extra>"))
    fig_tr.add_shape(type="line", x0=2019, x1=2019, y0=0, y1=1, yref="paper",
        line=dict(color="#e07b00",width=2,dash="dot"))
    fig_tr.add_shape(type="line", x0=2020, x1=2020, y0=0, y1=1, yref="paper",
        line=dict(color="#c62828",width=2,dash="dot"))
    fig_tr.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Economic Collapse 2019",
        line=dict(color="#e07b00",width=2,dash="dot"), showlegend=True))
    fig_tr.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="COVID + Blast 2020",
        line=dict(color="#c62828",width=2,dash="dot"), showlegend=True))
    fig_tr.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#1a1a1a", height=260,
        showlegend=True,
        legend=dict(bgcolor="rgba(255,255,255,0)", borderwidth=0, bordercolor="rgba(0,0,0,0)",
                    orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                    font=dict(color="#1a1a1a",size=8)),
        hovermode="x unified",
        margin=dict(l=50,r=70,t=55,b=30),
        xaxis=dict(showgrid=False, showline=True, linecolor="#cccccc", tickfont=dict(size=10), range=[2008.5,2023.5]),
        yaxis=dict(showgrid=False, showline=True, linecolor="#cccccc",
                   tickfont=dict(color="#2d6a4f",size=10),
                   title=dict(text="Prevalence (per 100k)", font=dict(color="#2d6a4f")), rangemode="tozero"),
        yaxis2=dict(showgrid=False, showline=True, linecolor="#cccccc",
                    tickfont=dict(color=eco_info["color"],size=10),
                    title=dict(text=eco_pick, font=dict(color=eco_info["color"])),
                    overlaying="y", side="right", rangemode="tozero"))
    st.plotly_chart(fig_tr, width="stretch")
 
# FORECAST
with r3b:
    st.markdown("<div class='sec'>Predictive Forecast (2024-2030) | Holt's Method | Lebanon</div>", unsafe_allow_html=True)
    hist_full   = df[(df.location=="Lebanon")&(df.cause==s_dc)&(df.measure==s_ms)&
                     (df.sex==s_sx)&(df.age=="All ages")&(df.metric=="Rate")&
                     (df.year>=1990)&(df.year<=2023)].sort_values("year")
    hist_recent = hist_full[hist_full.year>=2019].sort_values("year")
    n_steps = 7
    fy = np.arange(2024, 2031)
    if len(hist_recent) >= 3:
        series = hist_recent["val"].values
        model  = Holt(series, initialization_method="estimated").fit(optimized=True)
        yp     = model.forecast(n_steps)
        fitted = model.fittedvalues
        resid  = series[1:] - fitted[1:] if len(fitted)==len(series) else series - fitted
        std    = np.std(resid) if len(resid)>1 else series.std()*0.1
        hf     = np.sqrt(np.arange(1, n_steps+1))
        lower  = np.maximum(0, yp - 1.96*std*hf)
        upper  = yp + 1.96*std*hf
    else:
        X    = hist_recent["year"].values.reshape(-1,1)
        y_lr = hist_recent["val"].values
        mdl  = LinearRegression().fit(X, y_lr)
        yp   = mdl.predict(fy.reshape(-1,1))
        std  = np.std(y_lr - mdl.predict(X)) if len(y_lr)>2 else yp.std()*0.1
        lower = np.maximum(0, yp - 1.96*std)
        upper = yp + 1.96*std
    ff = go.Figure()
    ff.add_trace(go.Scatter(x=hist_full["year"], y=hist_full["val"], name="Historical (1990-2023)",
        line=dict(color="#2d6a4f",width=2), mode="lines+markers", marker=dict(size=3,color="#2d6a4f")))
    ff.add_trace(go.Scatter(x=hist_recent["year"], y=hist_recent["val"], name="Post-Crisis (2019-2023)",
        line=dict(color="#e07b00",width=2.5), mode="lines+markers", marker=dict(size=5,color="#e07b00")))
    ff.add_trace(go.Scatter(x=fy, y=yp, name="Forecast 2024-2030",
        line=dict(color="#c62828",width=2,dash="dash"),
        mode="lines+markers", marker=dict(size=5,symbol="diamond",color="#c62828")))
    ff.add_trace(go.Scatter(
        x=np.concatenate([fy, fy[::-1]]), y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(198,40,40,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    ff.add_vline(x=2019, line_dash="dot", line_color="#e07b00", line_width=1.5, annotation_text="")
    ff.add_vline(x=2023, line_dash="dot", line_color="#666666", line_width=1, annotation_text="")
    ff.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#1a1a1a", height=260,
        title=dict(text=f"{s_dcl} - {s_msl}", font=dict(size=11, color="#1a1a1a"), x=0),
        legend=dict(bgcolor="rgba(255,255,255,0)", borderwidth=0, bordercolor="rgba(0,0,0,0)",
                    font=dict(color="#1a1a1a",size=8),
                    orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        xaxis=dict(showgrid=False, showline=True, linecolor="#cccccc", tickfont=dict(size=9)),
        yaxis=dict(showgrid=False, showline=True, linecolor="#cccccc",
                   title="Rate per 100,000", tickfont=dict(size=9)),
        hovermode="x unified",
        margin=dict(l=50,r=10,t=55,b=30))
    st.plotly_chart(ff, width="stretch")
