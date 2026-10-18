import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & CUSTOM UI STYLING
# ==========================================
st.set_page_config(
    page_title="HER & OER Overpotential Analyzer",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS for bordered metric cards, clean tabs, and polished headers
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stFileUploader"] {
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Automated HER & OER Overpotential & EIS Analyzer")
st.caption(
    "Metrohm Autolab NOVA & Excel Template Processor • 100 kHz EIS Rs Extraction • "
    "90% iR Compensation • Nickel Foam Capacitive Baseline Diagnostics"
)

# ==========================================
# SIDEBAR: EXPERIMENTAL PARAMETERS
# ==========================================
with st.sidebar:
    st.header("⚙️ Experimental Parameters")

    pH = st.number_input("Electrolyte pH", min_value=0.0, max_value=14.5, value=14.0, step=0.1)

    ref_options = {
        "Ag/AgCl (3M KCl) [+0.197 V]": 0.197,
        "Ag/AgCl (Sat. KCl) [+0.197 V]": 0.197,
        "Hg/HgO (1M KOH) [+0.098 V]": 0.098,
        "SCE (Sat. Calomel) [+0.241 V]": 0.241,
        "Custom Reference": None,
    }
    ref_choice = st.selectbox("Reference Electrode", list(ref_options.keys()), index=0)
    if ref_options[ref_choice] is None:
        e_ref = st.number_input("Custom E°_ref (V vs. NHE)", value=0.197, format="%.4f")
    else:
        e_ref = ref_options[ref_choice]

    area_cm2 = st.number_input(
        "Geometric Electrode Area (cm²)", min_value=0.01, max_value=100.0, value=1.00, step=0.05
    )

    ir_comp_pct = st.slider(
        "iR Compensation (%)", min_value=0, max_value=100, value=90, step=5,
        help="Standard methodology uses 85–95% iR compensation (90% default)."
    )
    ir_factor = ir_comp_pct / 100.0

    rhe_offset = e_ref + (0.059 * pH)
    st.info(
        f"**RHE Conversion:**\n\n"
        f"$E_{{\\text{{RHE}}}} = E_{{\\text{{meas}}}} + {e_ref:.3f} + 0.059({pH:.1f})$\n\n"
        f"**Offset:** `+{rhe_offset:.4f} V`"
    )

    st.divider()
    st.header("🔧 Rs & Baseline Settings")
    rs_method = st.radio(
        "EIS Rs Extraction Method",
        [
            "Auto: Nearest to 100 kHz (Standard)",
            "Auto: Minimum |Z''| High-Freq Intercept",
            "Manual Rs Input",
        ],
        index=0,
    )
    manual_rs = st.number_input(
        "Manual / Fallback Rs (Ω)", min_value=0.0, max_value=5000.0, value=9.60, step=0.1,
        help="Used if Manual mode is selected or if no EIS file is uploaded."
    )

    apply_baseline = st.checkbox(
        "Apply Capacitive Pre-Current Baseline Zeroing",
        value=False,
        help="Subtracts non-Faradaic capacitive pre-current offset on high-surface-area Nickel Foam."
    )


# ==========================================
# CACHED HELPER FUNCTIONS
# ==========================================
@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes, file_name):
    if file_name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        return xls.sheet_names
    return ["Sheet1"]


@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes, file_name, sheet_name=None):
    name = file_name.lower()
    buf = io.BytesIO(file_bytes)
    if name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(buf)
        target = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=target)
    else:
        try:
            return pd.read_csv(buf, sep=None, engine="python")
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf)


def detect_column(df, keywords, fallback_idx=0):
    cols = list(df.columns)
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return cols[min(fallback_idx, len(cols) - 1)] if cols else None


def extract_rs_from_eis(df, col_freq, col_z_real, col_z_imag, method):
    work = df[[col_freq, col_z_real, col_z_imag]].apply(pd.to_numeric, errors="coerce").dropna()
    if work.empty:
        return None, None
    if "100 kHz" in method:
        idx = (work[col_freq] - 100000.0).abs().idxmin()
    else:
        high_freq_cut = work[col_freq].median()
        hf_sub = work[work[col_freq] >= high_freq_cut]
        if hf_sub.empty:
            hf_sub = work
        idx = hf_sub[col_z_imag].abs().idxmin()
    row = work.loc[idx]
    return float(row[col_z_real]), row


def process_lsv_data(
    df, col_pot, col_curr, curr_unit, reaction_type,
    rhe_offset_val, rs_ohm, ir_comp_frac, area, zero_baseline=False
):
    work = pd.DataFrame()
    work["WE(1).Potential (V)"] = pd.to_numeric(df[col_pot], errors="coerce")
    raw_curr = pd.to_numeric(df[col_curr], errors="coerce")
    work["WE(1).Current (A)"] = raw_curr / 1000.0 if curr_unit == "mA" else raw_curr
    work = work.dropna().reset_index(drop=True)

    baseline_offset_a = 0.0
    if zero_baseline and not work.empty:
        n_init = max(3, int(len(work) * 0.05))
        baseline_offset_a = work["WE(1).Current (A)"].iloc[:n_init].mean()
        work["Current_Corrected (A)"] = work["WE(1).Current (A)"] - baseline_offset_a
    else:
        work["Current_Corrected (A)"] = work["WE(1).Current (A)"]

    work["Current Density j (mA/cm²)"] = (work["Current_Corrected (A)"] * 1000.0) / area
    work["|j| (mA/cm²)"] = work["Current Density j (mA/cm²)"].abs()
    work["E vs RHE Raw (V)"] = work["WE(1).Potential (V)"] + rhe_offset_val
    work["iR Drop (V)"] = work["WE(1).Current (A)"] * rs_ohm * ir_comp_frac
    work["E vs RHE iR-Corrected (V)"] = work["E vs RHE Raw (V)"] - work["iR Drop (V)"]

    if reaction_type == "HER":
        work["Overpotential η Raw (mV)"] = (-work["E vs RHE Raw (V)"]) * 1000.0
        work["Overpotential η (mV)"] = (-work["E vs RHE iR-Corrected (V)"]) * 1000.0
    else:
        work["Overpotential η Raw (mV)"] = (work["E vs RHE Raw (V)"] - 1.23) * 1000.0
        work["Overpotential η (mV)"] = (work["E vs RHE iR-Corrected (V)"] - 1.23) * 1000.0

    work["log10(|j| [mA/cm²])"] = np.where(
        work["|j| (mA/cm²)"] > 1e-6, np.log10(work["|j| (mA/cm²)"].clip(lower=1e-6)), np.nan
    )
    return work, baseline_offset_a


def interpolate_eta_at_j(df_proc, target_j, use_ir_corrected=True):
    eta_col = "Overpotential η (mV)" if use_ir_corrected else "Overpotential η Raw (mV)"
    sub = df_proc[["|j| (mA/cm²)", eta_col]].dropna()
    if sub.empty or sub["|j| (mA/cm²)"].max() < target_j:
        return np.nan
    j_vals = sub["|j| (mA/cm²)"].values
    eta_vals = sub[eta_col].values
    for idx in range(len(j_vals) - 1):
        j1, j2 = j_vals[idx], j_vals[idx + 1]
        if (j1 <= target_j <= j2) or (j2 <= target_j <= j1):
            if abs(j2 - j1) < 1e-9:
                return float(eta_vals[idx])
            weight = (target_j - j1) / (j2 - j1)
            return float(eta_vals[idx] + weight * (eta_vals[idx + 1] - eta_vals[idx]))
    return np.nan


# ==========================================
# STEP 1: FILE UPLOADS
# ==========================================
st.subheader("1. Upload Raw Potentiostat Files (.xlsx, .csv, .txt)")
col_up1, col_up2, col_up3 = st.columns(3)

with col_up1:
    eis_file = st.file_uploader("Upload EIS Data (EIS.xlsx / .csv)", type=["xlsx", "xls", "csv", "txt"])
with col_up2:
    her_file = st.file_uploader("Upload LSV HER Data (LSV HER.xlsx / .csv)", type=["xlsx", "xls", "csv", "txt"])
with col_up3:
    oer_file = st.file_uploader("Upload LSV OER Data (LSV OER.xlsx / .csv)", type=["xlsx", "xls", "csv", "txt"])


# ==========================================
# STEP 2: EIS PROCESSING & Rs EXTRACTION
# ==========================================
active_rs = manual_rs
eis_df_clean = None

if eis_file is not None:
    st.divider()
    st.subheader("2. EIS Analysis & Solution Resistance ($R_s$)")
    eis_bytes = eis_file.getvalue()
    eis_sheets = get_sheet_names(eis_bytes, eis_file.name)

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    with c_s1:
        sel_eis_sheet = st.selectbox("EIS Sheet", eis_sheets) if len(eis_sheets) > 1 else eis_sheets[0]
    eis_raw = load_dataframe(eis_bytes, eis_file.name, sel_eis_sheet)

    with c_s2:
        col_freq = st.selectbox(
            "Frequency Column (Hz)",
            eis_raw.columns,
            index=list(eis_raw.columns).index(detect_column(eis_raw, ["freq", "hz"], 0)),
        )
    with c_s3:
        col_zr = st.selectbox(
            "Real Impedance Z' (Ω)",
            eis_raw.columns,
            index=list(eis_raw.columns).index(detect_column(eis_raw, ["z'", "zreal", "re(z)", "real"], 1)),
        )
    with c_s4:
        col_zi = st.selectbox(
            "Imaginary Impedance Z'' / -Z'' (Ω)",
            eis_raw.columns,
            index=list(eis_raw.columns).index(detect_column(eis_raw, ["-z''", "z''", "zimag", "im(z)", "imag"], 2)),
        )

    extracted_rs, rs_row = extract_rs_from_eis(eis_raw, col_freq, col_zr, col_zi, rs_method)
    if extracted_rs is not None and rs_method != "Manual Rs Input":
        active_rs = extracted_rs

    col_eis_metrics, col_eis_plot = st.columns([1, 2.2])
    with col_eis_metrics:
        st.metric("Extracted Solution Resistance (Rs)", f"{active_rs:.2f} Ω")
        if rs_row is not None:
            st.metric("Intercept Frequency", f"{rs_row[col_freq]:,.0f} Hz")
            st.caption(f"**Z':** `{rs_row[col_zr]:.3f} Ω` | **Z'':** `{rs_row[col_zi]:.3f} Ω`")
        st.success(f"**Effective iR Resistance ({ir_comp_pct}%):** `{active_rs * ir_factor:.2f} Ω`")

    with col_eis_plot:
        eis_df_clean = eis_raw[[col_freq, col_zr, col_zi]].apply(pd.to_numeric, errors="coerce").dropna()
        y_vals = -eis_df_clean[col_zi] if eis_df_clean[col_zi].mean() < 0 else eis_df_clean[col_zi]
        fig_eis = go.Figure()
        fig_eis.add_trace(
            go.Scatter(
                x=eis_df_clean[col_zr],
                y=y_vals,
                mode="lines+markers",
                name="Nyquist Curve",
                line=dict(color="#2563eb", width=2),
            )
        )
        if rs_row is not None:
            y_pt = -rs_row[col_zi] if eis_df_clean[col_zi].mean() < 0 else rs_row[col_zi]
            fig_eis.add_trace(
                go.Scatter(
                    x=[rs_row[col_zr]],
                    y=[y_pt],
                    mode="markers",
                    marker=dict(size=13, color="#dc2626", symbol="x"),
                    name=f"Rs = {active_rs:.2f} Ω (@ {rs_row[col_freq]:.0f} Hz)",
                )
            )
        fig_eis.update_layout(
            template="plotly_white",
            title="EIS Nyquist Plot & High-Frequency Rs Intercept",
            xaxis_title="Z' (Ω)",
            yaxis_title="-Z'' (Ω)",
            height=330,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_eis, use_container_width=True)
else:
    st.info(f"💡 No EIS file uploaded yet. Using fallback/manual **$R_s = {active_rs:.2f}\\ \\Omega$** from the sidebar.")


# ==========================================
# FRAGMENT FOR ZERO-LAG TAFEL & PLOTTING
# ==========================================
@st.fragment
def render_interactive_plots(df_proc, reaction_label, ir_comp_pct_val):
    valid_tafel = df_proc.dropna(subset=["log10(|j| [mA/cm²])", "Overpotential η (mV)"])
    valid_tafel = valid_tafel[valid_tafel["Overpotential η (mV)"] > 0]

    col_t1, col_t2 = st.columns([1, 2.2])
    tafel_slope = np.nan
    tafel_r2 = np.nan

    with col_t1:
        st.markdown(f"##### 🎯 {reaction_label} Tafel Linear Fit")
        if len(valid_tafel) >= 5:
            j_min_avail = float(max(0.5, valid_tafel["|j| (mA/cm²)"].min()))
            j_max_avail = float(valid_tafel["|j| (mA/cm²)"].max())
            default_low = min(max(5.0, j_min_avail), j_max_avail * 0.5)
            default_high = min(max(30.0, default_low + 10.0), j_max_avail)

            j_range = st.slider(
                f"Linear Window |j| (mA/cm²)",
                min_value=round(j_min_avail, 2),
                max_value=round(j_max_avail, 2),
                value=(round(default_low, 2), round(default_high, 2)),
                key=f"tafel_slider_{reaction_label}",
            )

            fit_sub = valid_tafel[
                (valid_tafel["|j| (mA/cm²)"] >= j_range[0])
                & (valid_tafel["|j| (mA/cm²)"] <= j_range[1])
            ]

            if len(fit_sub) >= 3:
                res = linregress(fit_sub["log10(|j| [mA/cm²])"], fit_sub["Overpotential η (mV)"])
                tafel_slope = res.slope
                tafel_r2 = res.rvalue ** 2
                st.metric(f"{reaction_label} Tafel Slope", f"{tafel_slope:.1f} mV/dec")
                st.metric("Linear Fit R²", f"{tafel_r2:.5f}")
            else:
                st.warning("Select a wider current density window (≥ 3 points).")

    with col_t2:
        tab_lsv, tab_tafel, tab_table = st.tabs(
            ["📈 LSV Polarization Curve", "📐 Tafel Plot (η vs. log|j|)", "📋 Processed Data Table"]
        )

        with tab_lsv:
            fig_lsv = go.Figure()
            fig_lsv.add_trace(
                go.Scatter(
                    x=df_proc["E vs RHE Raw (V)"],
                    y=df_proc["Current Density j (mA/cm²)"],
                    mode="lines",
                    name="Raw (0% iR)",
                    line=dict(color="#94a3b8", dash="dash", width=2),
                )
            )
            fig_lsv.add_trace(
                go.Scatter(
                    x=df_proc["E vs RHE iR-Corrected (V)"],
                    y=df_proc["Current Density j (mA/cm²)"],
                    mode="lines",
                    name=f"{ir_comp_pct_val}% iR-Corrected",
                    line=dict(color="#0f766e" if reaction_label == "HER" else "#b91c1c", width=3),
                )
            )
            fig_lsv.update_layout(
                template="plotly_white",
                xaxis_title="Potential (V vs. RHE)",
                yaxis_title="Current Density j (mA/cm²)",
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_lsv, use_container_width=True)

        with tab_tafel:
            fig_tf = go.Figure()
            fig_tf.add_trace(
                go.Scatter(
                    x=valid_tafel["log10(|j| [mA/cm²])"],
                    y=valid_tafel["Overpotential η (mV)"],
                    mode="markers",
                    name="Experimental Data",
                    marker=dict(color="#64748b", size=6, opacity=0.7),
                )
            )
            if not np.isnan(tafel_slope):
                x_fit = fit_sub["log10(|j| [mA/cm²])"]
                y_fit = res.intercept + res.slope * x_fit
                fig_tf.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        line=dict(color="#dc2626", width=3),
                        name=f"Fit: {tafel_slope:.1f} mV/dec (R²={tafel_r2:.4f})",
                    )
                )
            fig_tf.update_layout(
                template="plotly_white",
                xaxis_title="log₁₀(|j| / mA cm⁻²)",
                yaxis_title="Overpotential η (mV)",
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_tf, use_container_width=True)

        with tab_table:
            st.dataframe(df_proc, use_container_width=True, height=330)

    return tafel_slope, tafel_r2


# ==========================================
# STEP 3: LSV HER & OER PROCESSING
# ==========================================
def render_reaction_section(uploaded_lsv, reaction_label):
    st.divider()
    st.subheader(f"📊 {reaction_label} Overpotential & Tafel Analysis")

    lsv_bytes = uploaded_lsv.getvalue()
    lsv_sheets = get_sheet_names(lsv_bytes, uploaded_lsv.name)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if len(lsv_sheets) > 1:
            sel_sheet = st.selectbox(f"{reaction_label} Sheet", lsv_sheets, key=f"sheet_{reaction_label}")
        else:
            sel_sheet = lsv_sheets[0]
            st.text_input(f"{reaction_label} Sheet", value=sel_sheet, disabled=True, key=f"sheet_{reaction_label}")

    lsv_raw = load_dataframe(lsv_bytes, uploaded_lsv.name, sel_sheet)

    with c2:
        col_pot = st.selectbox(
            f"{reaction_label} Potential Column",
            lsv_raw.columns,
            index=list(lsv_raw.columns).index(detect_column(lsv_raw, ["WE(1).Potential", "potential", "v"], 0)),
            key=f"pot_{reaction_label}",
        )
    with c3:
        col_curr = st.selectbox(
            f"{reaction_label} Current Column",
            lsv_raw.columns,
            index=list(lsv_raw.columns).index(detect_column(lsv_raw, ["WE(1).Current", "current", "i", "a"], 1)),
            key=f"curr_{reaction_label}",
        )
    with c4:
        curr_unit = st.selectbox(
            f"{reaction_label} Raw Current Unit",
            ["A (Autolab NOVA Default)", "mA"],
            index=0,
            key=f"unit_{reaction_label}",
        )
        unit_flag = "mA" if curr_unit == "mA" else "A"

    df_proc, base_offset = process_lsv_data(
        lsv_raw, col_pot, col_curr, unit_flag, reaction_label,
        rhe_offset, active_rs, ir_factor, area_cm2, zero_baseline=apply_baseline,
    )

    benchmarks = [10, 20, 30, 50, 100]
    eta_corr_dict = {j_b: interpolate_eta_at_j(df_proc, j_b, True) for j_b in benchmarks}
    eta_raw_dict = {j_b: interpolate_eta_at_j(df_proc, j_b, False) for j_b in benchmarks}

    init_j = df_proc["|j| (mA/cm²)"].iloc[: max(3, int(len(df_proc) * 0.05))].mean()
    if init_j > 3.0 and not apply_baseline:
        st.warning(
            f"⚠️ **Capacitive / Redox Pre-Current Detected (`{init_j:.2f} mA/cm²`):** "
            f"High-surface-area Nickel Foam pre-currents can artificially skew $\\eta_{{10}}$. "
            f"Compare **$\\eta_{{20}}$, $\\eta_{{30}}$, and $\\eta_{{50}}$** below, or enable **Capacitive Baseline Zeroing** in the sidebar."
        )
    elif apply_baseline:
        st.success(f"✔️ **Baseline Zeroing Active:** Subtracted `{base_offset * 1000:.3f} mA` non-Faradaic pre-current.")

    m_cols = st.columns(len(benchmarks))
    for idx, j_b in enumerate(benchmarks):
        val_c = eta_corr_dict[j_b]
        val_r = eta_raw_dict[j_b]
        with m_cols[idx]:
            st.metric(
                label=f"η_{j_b} ({ir_comp_pct}% iR)",
                value=f"{val_c:.1f} mV" if not np.isnan(val_c) else "N/A",
                delta=f"Raw (0% iR): {val_r:.1f} mV" if not np.isnan(val_r) else None,
                delta_color="off",
            )

    tafel_slope, tafel_r2 = render_interactive_plots(df_proc, reaction_label, ir_comp_pct)

    summary_info = {
        "Reaction": reaction_label,
        "Rs (Ω)": active_rs,
        "iR Compensation (%)": ir_comp_pct,
        "pH": pH,
        "RHE Offset (V)": rhe_offset,
        "Area (cm²)": area_cm2,
        "η10 (mV) [iR-Corr]": eta_corr_dict[10],
        "η20 (mV) [iR-Corr]": eta_corr_dict[20],
        "η30 (mV) [iR-Corr]": eta_corr_dict[30],
        "η50 (mV) [iR-Corr]": eta_corr_dict[50],
        "η100 (mV) [iR-Corr]": eta_corr_dict[100],
        "η10 (mV) [Raw]": eta_raw_dict[10],
        "η50 (mV) [Raw]": eta_raw_dict[50],
        "Tafel Slope (mV/dec)": tafel_slope,
        "Tafel R²": tafel_r2,
    }
    return df_proc, summary_info


her_processed, her_summary = (None, None)
oer_processed, oer_summary = (None, None)

if her_file is not None:
    her_processed, her_summary = render_reaction_section(her_file, "HER")

if oer_file is not None:
    oer_processed, oer_summary = render_reaction_section(oer_file, "OER")


# ==========================================
# STEP 4: ONE-CLICK EXCEL WORKBOOK EXPORT
# ==========================================
if her_processed is not None or oer_processed is not None:
    st.divider()
    st.subheader("📥 Download Processed Excel Workbook")

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        summaries = [s for s in [her_summary, oer_summary] if s is not None]
        pd.DataFrame(summaries).to_excel(writer, sheet_name="Summary_Metrics", index=False)

        if eis_df_clean is not None:
            eis_df_clean.to_excel(writer, sheet_name="EIS_Data", index=False)
        if her_processed is not None:
            her_processed.to_excel(writer, sheet_name="LSV_HER_Calculated", index=False)
        if oer_processed is not None:
            oer_processed.to_excel(writer, sheet_name="LSV_OER_Calculated", index=False)

    st.download_button(
        label="📊 Download Complete HER/OER Calculation Workbook (.xlsx)",
        data=output_buffer.getvalue(),
        file_name="Processed_HER_OER_Overpotential_Calculations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )