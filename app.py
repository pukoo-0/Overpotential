import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress
import streamlit as st

st.set_page_config(page_title="Electrochemistry Calculator", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1300px; }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 6px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_sheets(file_bytes, file_name):
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    return ["Sheet1"]


@st.cache_data(show_spinner=False)
def read_data(file_bytes, file_name, sheet_name=None):
    buf = io.BytesIO(file_bytes)
    if file_name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(buf)
        target = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=target)
    try:
        return pd.read_csv(buf, sep=None, engine="python")
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf)


def find_col(df, keywords, default_idx=0):
    cols = list(df.columns)
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return cols[min(default_idx, len(cols) - 1)] if cols else None


def get_rs(df, col_f, col_zr, col_zi, mode):
    sub = df[[col_f, col_zr, col_zi]].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        return None, None
    if mode == "100 kHz":
        idx = (sub[col_f] - 100000.0).abs().idxmin()
    else:
        hf = sub[sub[col_f] >= sub[col_f].median()]
        if hf.empty:
            hf = sub
        idx = hf[col_zi].abs().idxmin()
    row = sub.loc[idx]
    return float(row[col_zr]), row


def calc_lsv(df, col_v, col_i, i_unit, rxn, rhe_off, rs, ir_frac, area):
    out = pd.DataFrame()
    out["Potential (V)"] = pd.to_numeric(df[col_v], errors="coerce")
    raw_i = pd.to_numeric(df[col_i], errors="coerce")
    out["Current (A)"] = raw_i / 1000.0 if i_unit == "mA" else raw_i
    out = out.dropna().reset_index(drop=True)

    out["j (mA/cm²)"] = (out["Current (A)"] * 1000.0) / area
    out["|j| (mA/cm²)"] = out["j (mA/cm²)"].abs()
    out["E_RHE Raw (V)"] = out["Potential (V)"] + rhe_off
    out["iR Drop (V)"] = out["Current (A)"] * rs * ir_frac
    out["E_RHE (V)"] = out["E_RHE Raw (V)"] - out["iR Drop (V)"]

    if rxn == "HER":
        out["η Raw (mV)"] = (-out["E_RHE Raw (V)"]) * 1000.0
        out["η (mV)"] = (-out["E_RHE (V)"]) * 1000.0
    else:
        out["η Raw (mV)"] = (out["E_RHE Raw (V)"] - 1.23) * 1000.0
        out["η (mV)"] = (out["E_RHE (V)"] - 1.23) * 1000.0

    out["log|j|"] = np.where(
        out["|j| (mA/cm²)"] > 1e-6, np.log10(out["|j| (mA/cm²)"].clip(lower=1e-6)), np.nan
    )
    return out


def get_eta10(df, eta_col="η (mV)", e_col="E_RHE (V)"):
    """
    Finds overpotential at 10 mA/cm² on the rising catalytic branch only,
    ignoring any initial potentiostat settling spike on the left.
    """
    sub = df[["|j| (mA/cm²)", eta_col, e_col]].dropna().reset_index(drop=True)
    if sub.empty:
        return np.nan, np.nan

    # Start searching only after the baseline minimum current density
    min_idx = int(sub["|j| (mA/cm²)"].idxmin())
    rising_branch = sub.iloc[min_idx:].reset_index(drop=True)

    jv = rising_branch["|j| (mA/cm²)"].values
    ev = rising_branch[eta_col].values
    vv = rising_branch[e_col].values

    for k in range(len(jv) - 1):
        # Require rising current density crossing 10 mA/cm² and positive overpotential
        if jv[k] <= 10.0 <= jv[k + 1] and ev[k + 1] > 0:
            if abs(jv[k + 1] - jv[k]) < 1e-9:
                return float(ev[k]), float(vv[k])
            w = (10.0 - jv[k]) / (jv[k + 1] - jv[k])
            eta_10 = float(ev[k] + w * (ev[k + 1] - ev[k]))
            v_10 = float(vv[k] + w * (vv[k + 1] - vv[k]))
            return eta_10, v_10

    return np.nan, np.nan


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.subheader("Parameters")
    pH = st.number_input("pH", 0.0, 14.5, 14.0, 0.1)

    refs = {
        "Ag/AgCl (0.197 V)": 0.197,
        "Hg/HgO (0.098 V)": 0.098,
        "SCE (0.241 V)": 0.241,
        "Custom": None,
    }
    ref_sel = st.selectbox("Reference", list(refs.keys()))
    e_ref = st.number_input("E°_ref (V)", value=0.197, format="%.4f") if refs[ref_sel] is None else refs[ref_sel]

    area_cm2 = st.number_input("Area (cm²)", 0.01, 100.0, 1.00, 0.05)
    ir_pct = st.number_input("iR Compensation (%)", 0, 100, 90, 5)
    ir_frac = ir_pct / 100.0

    st.divider()
    st.subheader("Resistance (Rs)")
    rs_mode = st.selectbox("Rs Source", ["100 kHz", "Min |Z''|", "Manual"])
    manual_rs = st.number_input("Manual Rs (Ω)", 0.0, 5000.0, 9.60, 0.1)

    rhe_offset = e_ref + 0.059 * pH
    st.caption(f"RHE offset: +{rhe_offset:.4f} V")


# ==========================================
# MAIN HEADER & FILE UPLOADS
# ==========================================
st.subheader("Overpotential & Tafel Calculator")

u1, u2, u3 = st.columns(3)
with u1:
    eis_file = st.file_uploader("EIS", type=["xlsx", "xls", "csv", "txt"])
with u2:
    her_file = st.file_uploader("HER LSV", type=["xlsx", "xls", "csv", "txt"])
with u3:
    oer_file = st.file_uploader("OER LSV", type=["xlsx", "xls", "csv", "txt"])


# ==========================================
# EIS EXTRACTION
# ==========================================
active_rs = manual_rs
eis_clean = None
eis_fig = None

if eis_file is not None:
    eis_bytes = eis_file.getvalue()
    eis_sheets = get_sheets(eis_bytes, eis_file.name)
    eis_raw = read_data(eis_bytes, eis_file.name, eis_sheets[0])

    cf = find_col(eis_raw, ["freq", "hz"], 0)
    czr = find_col(eis_raw, ["z'", "zreal", "re(z)", "real"], 1)
    czi = find_col(eis_raw, ["-z''", "z''", "zimag", "im(z)", "imag"], 2)

    ext_rs, rs_pt = get_rs(eis_raw, cf, czr, czi, rs_mode)
    if ext_rs is not None and rs_mode != "Manual":
        active_rs = ext_rs

    eis_clean = eis_raw[[cf, czr, czi]].apply(pd.to_numeric, errors="coerce").dropna()
    yv = -eis_clean[czi] if eis_clean[czi].mean() < 0 else eis_clean[czi]

    eis_fig = go.Figure()
    eis_fig.add_trace(go.Scatter(x=eis_clean[czr], y=yv, mode="lines+markers", name="EIS"))
    if rs_pt is not None:
        y_p = -rs_pt[czi] if eis_clean[czi].mean() < 0 else rs_pt[czi]
        eis_fig.add_trace(
            go.Scatter(
                x=[rs_pt[czr]],
                y=[y_p],
                mode="markers",
                marker=dict(size=10, color="red", symbol="x"),
                name=f"Rs = {active_rs:.2f} Ω",
            )
        )
    eis_fig.update_layout(
        template="plotly_white",
        xaxis_title="Z' (Ω)",
        yaxis_title="-Z'' (Ω)",
        height=340,
        margin=dict(l=10, r=10, t=25, b=10),
    )


# ==========================================
# REACTION VIEW
# ==========================================
@st.fragment
def render_rxn(uploaded_file, rxn):
    raw_bytes = uploaded_file.getvalue()
    sheets = get_sheets(raw_bytes, uploaded_file.name)

    with st.expander("Column & Sheet Mapping", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            sh = st.selectbox("Sheet", sheets, key=f"sh_{rxn}")
        df_raw = read_data(raw_bytes, uploaded_file.name, sh)
        with m2:
            cv = st.selectbox(
                "Potential",
                df_raw.columns,
                index=list(df_raw.columns).index(find_col(df_raw, ["WE(1).Potential", "potential", "v"], 0)),
                key=f"cv_{rxn}",
            )
        with m3:
            ci = st.selectbox(
                "Current",
                df_raw.columns,
                index=list(df_raw.columns).index(find_col(df_raw, ["WE(1).Current", "current", "i", "a"], 1)),
                key=f"ci_{rxn}",
            )
        with m4:
            iu = st.selectbox("Unit", ["A", "mA"], index=0, key=f"iu_{rxn}")

    df_proc = calc_lsv(df_raw, cv, ci, iu, rxn, rhe_offset, active_rs, ir_frac, area_cm2)

    # Extract η10 on rising catalytic branch only
    eta10, v10 = get_eta10(df_proc, "η (mV)", "E_RHE (V)")
    eta10_raw, _ = get_eta10(df_proc, "η Raw (mV)", "E_RHE Raw (V)")

    # Slice rising catalytic branch with η > 0 for Tafel fitting
    min_idx = int(df_proc["|j| (mA/cm²)"].idxmin())
    tf_data = df_proc.iloc[min_idx:].dropna(subset=["log|j|", "η (mV)"])
    tf_data = tf_data[tf_data["η (mV)"] > 0]

    j_min = float(max(0.5, tf_data["|j| (mA/cm²)"].min())) if len(tf_data) >= 5 else 1.0
    j_max = float(tf_data["|j| (mA/cm²)"].max()) if len(tf_data) >= 5 else 100.0
    d_low = min(max(5.0, j_min), j_max * 0.5)
    d_high = min(max(30.0, d_low + 10.0), j_max)

    cols = st.columns(4)
    cols[0].metric("η10", f"{eta10:.1f} mV" if not np.isnan(eta10) else "—")
    cols[1].metric("E @ 10 mA/cm²", f"{v10:.3f} V" if not np.isnan(v10) else "—")
    cols[2].metric("Rs", f"{active_rs:.2f} Ω")

    p1, p2 = st.columns(2)

    with p2:
        j_win = st.slider(
            "Tafel Fit Window (mA/cm²)",
            min_value=round(j_min, 1),
            max_value=round(j_max, 1),
            value=(round(d_low, 1), round(d_high, 1)),
            key=f"tf_{rxn}",
        )

    slope, r2 = np.nan, np.nan
    fit_sub = tf_data[(tf_data["|j| (mA/cm²)"] >= j_win[0]) & (tf_data["|j| (mA/cm²)"] <= j_win[1])]
    if len(fit_sub) >= 3:
        reg = linregress(fit_sub["log|j|"], fit_sub["η (mV)"])
        slope, r2 = reg.slope, reg.rvalue ** 2

    cols[3].metric("Tafel Slope", f"{slope:.1f} mV/dec" if not np.isnan(slope) else "—")

    with p1:
        fig_lsv = go.Figure()
        fig_lsv.add_trace(
            go.Scatter(
                x=df_proc["E_RHE Raw (V)"],
                y=df_proc["j (mA/cm²)"],
                mode="lines",
                name="Raw",
                line=dict(color="#94a3b8", dash="dash"),
            )
        )
        fig_lsv.add_trace(
            go.Scatter(
                x=df_proc["E_RHE (V)"],
                y=df_proc["j (mA/cm²)"],
                mode="lines",
                name=f"{ir_pct}% iR",
                line=dict(color="#111827", width=2.5),
            )
        )
        if not np.isnan(v10):
            j_target_signed = -10.0 if rxn == "HER" else 10.0
            fig_lsv.add_trace(
                go.Scatter(
                    x=[v10],
                    y=[j_target_signed],
                    mode="markers",
                    marker=dict(size=10, color="#dc2626"),
                    name=f"η10 = {eta10:.1f} mV",
                )
            )
        fig_lsv.update_layout(
            template="plotly_white",
            title="LSV Curve",
            xaxis_title="E (V vs. RHE)",
            yaxis_title="j (mA/cm²)",
            height=350,
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(fig_lsv, use_container_width=True)

    with p2:
        fig_tf = go.Figure()
        fig_tf.add_trace(
            go.Scatter(
                x=tf_data["log|j|"],
                y=tf_data["η (mV)"],
                mode="markers",
                name="Data",
                marker=dict(color="#94a3b8", size=5),
            )
        )
        if not np.isnan(slope):
            xf = fit_sub["log|j|"]
            yf = reg.intercept + reg.slope * xf
            fig_tf.add_trace(
                go.Scatter(
                    x=xf,
                    y=yf,
                    mode="lines",
                    name=f"{slope:.1f} mV/dec (R²={r2:.4f})",
                    line=dict(color="#dc2626", width=2.5),
                )
            )
        fig_tf.update_layout(
            template="plotly_white",
            title=f"Tafel Plot (R² = {r2:.4f})" if not np.isnan(r2) else "Tafel Plot",
            xaxis_title="log |j| (mA/cm²)",
            yaxis_title="η (mV)",
            height=300,
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(fig_tf, use_container_width=True)

    with st.expander("View Processed Table", expanded=False):
        st.dataframe(df_proc, use_container_width=True, height=260)

    summary = {
        "Reaction": rxn,
        "Rs (Ω)": active_rs,
        "iR (%)": ir_pct,
        "pH": pH,
        "Area (cm²)": area_cm2,
        "E @ 10 mA/cm² (V vs RHE)": v10,
        "η10 (mV)": eta10,
        "η10 Raw (mV)": eta10_raw,
        "Tafel Slope (mV/dec)": slope,
        "Tafel R²": r2,
    }
    return df_proc, summary


# ==========================================
# TABS FOR UPLOADED DATA
# ==========================================
active_tabs = []
if her_file:
    active_tabs.append("HER")
if oer_file:
    active_tabs.append("OER")
if eis_file:
    active_tabs.append("EIS")

her_df, her_sum = None, None
oer_df, oer_sum = None, None

if active_tabs:
    st.divider()
    tabs = st.tabs(active_tabs)
    for i, t_name in enumerate(active_tabs):
        with tabs[i]:
            if t_name == "HER":
                her_df, her_sum = render_rxn(her_file, "HER")
            elif t_name == "OER":
                oer_df, oer_sum = render_rxn(oer_file, "OER")
            elif t_name == "EIS":
                c_m, c_p = st.columns([1, 3])
                c_m.metric("Rs", f"{active_rs:.2f} Ω")
                c_m.metric("Compensated Rs", f"{active_rs * ir_frac:.2f} Ω")
                c_p.plotly_chart(eis_fig, use_container_width=True)

    if her_df is not None or oer_df is not None:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame([s for s in [her_sum, oer_sum] if s]).to_excel(writer, sheet_name="Summary", index=False)
            if eis_clean is not None:
                eis_clean.to_excel(writer, sheet_name="EIS", index=False)
            if her_df is not None:
                her_df.to_excel(writer, sheet_name="HER", index=False)
            if oer_df is not None:
                oer_df.to_excel(writer, sheet_name="OER", index=False)

        st.download_button(
            label="Export Excel (.xlsx)",
            data=buf.getvalue(),
            file_name="Overpotential_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )