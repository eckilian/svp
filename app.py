# ============================================================
# SVP Analyzer – Streamlit App (mit Berechnen-Button, What-If Fix)
# - Datei-Upload (xlsx/csv, optional Blatt-Hinweis)
# - Auto-Erkennung: Faktoren/Ergebnisse
# - Stufen-Filter je Faktor
# - "🚀 Berechnen" Button -> nichts rechnet vorher
# - Analysen: ANOVA (Typ II/III), Haupteffekte, Interaktionen, Boxplots,
#   Residuen, Daniel-Plot, 3D (Surface/Scatter + Heatmap + Export), PCA (optional)
# - What-If: Vorhersage (95%-CI + Vorhersageintervall) – gefixt & robust
# - Downloads: alle Plots als SVG (Direkt-Download + ZIP); Tabellen als Excel
# - Sichere Excel-Sheetnamen
# ============================================================

import io, zipfile, re, warnings, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import seaborn as sns
import streamlit as st

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from scipy.stats import halfnorm, t as t_dist

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_style("whitegrid")


# -------------------------------
# Page / Sidebar
# -------------------------------
st.set_page_config(page_title="SVP Analyzer", layout="wide")
st.title("Statistische Versuchsplanung Auswertung")

st.sidebar.header("⚙️ Optionen")
unique_threshold = st.sidebar.slider("Schwelle: ≤ X eindeutige Werte ⇒ Faktor", 2, 30, 10, 1)
sheet_hint = st.sidebar.text_input("Excel-Blatthinweis (optional, leer=erstes Blatt)", value="")

st.sidebar.markdown("**Analysen auswählen:**")
opt_anova  = st.sidebar.checkbox("ANOVA (Type II/III) + Effektstärken", True)
opt_main   = st.sidebar.checkbox("Haupteffekt‑Plots (mit SE)", True)
opt_inter  = st.sidebar.checkbox("Interaktions‑Plots (mit SE)", True)
opt_box    = st.sidebar.checkbox("Boxplots (≥3 Stufen)", True)
opt_resid  = st.sidebar.checkbox("Residuen‑Plots", True)
opt_daniel = st.sidebar.checkbox("Daniel‑Plot (2‑Stufen‑Faktoren)", True)
opt_3d     = st.sidebar.checkbox("3D‑Plot (2 Faktoren)", True)
opt_pca    = st.sidebar.checkbox("PCA (Scree, Loadings, Biplot)", False)
opt_whatif = st.sidebar.checkbox("What‑If (Vorhersage, 95%-CI)", True)
opt_zip    = st.sidebar.checkbox("ZIP‑Download aller SVG‑Grafiken & Tabellen", True)


# -------------------------------
# File uploader
# -------------------------------
uploaded = st.file_uploader("📂 Excel/CSV hochladen", type=["xlsx", "csv"])
if uploaded is None:
    st.stop()

def _file_sig(uploaded_file) -> str:
    try:
        data = uploaded_file.getvalue()
    except Exception:
        data = uploaded_file.read()
    return hashlib.md5(data).hexdigest()


# -------------------------------
# Robust file read
# -------------------------------
try:
    if uploaded.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        xls = pd.ExcelFile(uploaded)
        if sheet_hint.strip() and sheet_hint in xls.sheet_names:
            df_raw = xls.parse(sheet_hint)
        else:
            df_raw = xls.parse(xls.sheet_names[0])
except Exception as e:
    st.error(f"❌ Fehler beim Lesen der Datei: {e}")
    st.stop()

st.success("✔ Datei gelesen")
st.write("### Vorschau")
st.dataframe(df_raw.head())


# -------------------------------
# Helper: detection & safe naming
# -------------------------------
all_cols = list(df_raw.columns)

def nonempty_unique(series: pd.Series) -> int:
    return int(series.dropna().nunique())

# Faktoren = nicht-numerisch ODER ≤ threshold (aber nur, wenn ≥1 gültiger Wert)
auto_factors = []
for c in all_cols:
    nun = nonempty_unique(df_raw[c])
    if nun == 0:
        continue
    if (not np.issubdtype(df_raw[c].dtype, np.number)) or nun <= unique_threshold:
        auto_factors.append(c)

# Ergebnisse = numerisch, nicht Faktor
auto_results = [
    c for c in all_cols
    if np.issubdtype(df_raw[c].dtype, np.number) and c not in auto_factors
]

# User selection: factors / results
st.header("1️⃣ Faktoren & Ergebnisse auswählen")
factors = st.multiselect("Faktoren wählen:", options=all_cols, default=auto_factors)
results = st.multiselect(
    "Ergebnisse (numerische Qualitäten) wählen:",
    options=[c for c in all_cols if np.issubdtype(df_raw[c].dtype, np.number)],
    default=auto_results
)
if len(results) == 0:
    st.error("❌ Es muss mindestens **eine** Ergebnisspalte ausgewählt werden.")
    st.stop()

# cast factors to category, leere Faktoren verwerfend
df = df_raw.copy()
clean_factors = []
for colF in factors:
    if nonempty_unique(df[colF]) == 0:
        st.warning(f"⚠ Faktor '{colF}' hat 0 gültige Werte ⇒ übersprungen.")
        continue
    df[colF] = df[colF].astype("category")
    clean_factors.append(colF)

# sichere Spaltennamen für Formeln
def make_safe_names(cols, prefix):
    safe_map = {}
    used = set()
    for i, c in enumerate(cols, 1):
        base = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        if not base or (base and base[0].isdigit()):
            base = f"{prefix}{i}"
        name = base
        k = 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        safe_map[c] = name
        used.add(name)
    return safe_map

fac_map = make_safe_names(clean_factors, "F")
res_map = make_safe_names(results, "Y")

df_safe = df.rename(columns={**fac_map, **res_map})
safe_factors = [fac_map[f] for f in clean_factors]
safe_results = [res_map[r] for r in results]


# -------------------------------
# Level filter per factor
# -------------------------------
st.header("2️⃣ Stufen je Faktor auswählen (optional filtern)")
df_work = df_safe.copy()
for F in safe_factors:
    origF = list(fac_map.keys())[safe_factors.index(F)]
    levels_all = list(df_work[F].astype("category").cat.categories)
    levels_sel = st.multiselect(
        f"Stufen in **{origF}** verwenden:",
        options=levels_all,
        default=levels_all,
        key=f"lv_{F}"
    )
    df_work = df_work[df_work[F].isin(levels_sel)]
    df_work[F] = df_work[F].astype("category").cat.remove_unused_categories()

if df_work.empty:
    st.error("❌ Nach dem Filtern enthält der Datensatz keine Zeilen mehr.")
    st.stop()


# -------------------------------
# 3D Auswahl & What-If Ziel (vor Berechnen auswählen)
# -------------------------------
f1_3d = f2_3d = None
if opt_3d and len(safe_factors) >= 2:
    col_3d_1, col_3d_2 = st.columns(2)
    with col_3d_1:
        f1_3d = st.selectbox(
            "3D‑Faktor X",
            options=safe_factors,
            index=0,
            format_func=lambda s: list(fac_map.keys())[safe_factors.index(s)]
        )
    with col_3d_2:
        f2_3d = st.selectbox(
            "3D‑Faktor Y",
            options=[f for f in safe_factors if f != f1_3d],
            index=0,
            format_func=lambda s: list(fac_map.keys())[safe_factors.index(s)]
        )

# What‑If Response Auswahl (nur Zielvariable; Levels kommen später)
whatif_y = None
if opt_whatif and safe_results:
    whatif_y = st.selectbox(
        "What‑If – Zielgröße wählen",
        options=safe_results,
        index=0,
        format_func=lambda s: list(res_map.keys())[safe_results.index(s)]
    )


# -------------------------------
# Berechnen-Button & State
# -------------------------------
# Init State
if "computed" not in st.session_state:
    st.session_state.computed = False
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# Hilfsobjekt zur Änderungserkennung (ohne What-If-Level, damit What-If interaktiv bleiben kann)
current_levels = {
    F: st.session_state.get(f"lv_{F}", None) for F in safe_factors
}
current_inputs = {
    "file_sig": _file_sig(uploaded),
    "unique_threshold": unique_threshold,
    "sheet_hint": sheet_hint,
    "factors": list(factors),
    "results": list(results),
    "safe_factors": list(safe_factors),
    "safe_results": list(safe_results),
    "levels": current_levels,
    "opts": {
        "anova": opt_anova, "main": opt_main, "inter": opt_inter, "box": opt_box,
        "resid": opt_resid, "daniel": opt_daniel, "pca": opt_pca, "do3d": opt_3d
    },
    "f1_3d": f1_3d, "f2_3d": f2_3d,
}

ready_to_run = (uploaded is not None) and (len(results) > 0) and (not df_work.empty)
col_run1, col_run2 = st.columns([1, 3])
with col_run1:
    run_clicked = st.button("🚀 Berechnen", type="primary", disabled=not ready_to_run)

# Inputs geändert?
inputs_changed = (st.session_state.last_inputs is not None) and (st.session_state.last_inputs != current_inputs)

# State-Übergang
if run_clicked:
    st.session_state.computed = True
    st.session_state.last_inputs = current_inputs

if not st.session_state.computed:
    st.info("⬆️ Wähle Datei/Faktoren/Ergebnisse/Stufen (optional 3D & Ziel für What‑If) und klicke **Berechnen**.")
    st.stop()

if inputs_changed and not run_clicked:
    st.warning("⚠️ Auswahl geändert – Ergebnisse sind veraltet. Bitte erneut **Berechnen**.")
    st.stop()


# -------------------------------
# ZIP – Inhalte sammeln und am Ende packen
# -------------------------------
zip_contents: dict[str, bytes] = {}   # pfad → bytes

def _to_bytes(obj) -> bytes:
    if isinstance(obj, bytes):
        return obj
    return str(obj).encode("utf-8")

def _register(path: str, data):
    if opt_zip:
        zip_contents[path] = _to_bytes(data)

def _register_df_csv(path: str, df_in: pd.DataFrame):
    if opt_zip:
        zip_contents[path] = df_in.to_csv(index=False).encode("utf-8-sig")

def fig_to_svg(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def add_svg(path: str, fig):
    if opt_zip:
        zip_contents[path] = fig_to_svg(fig)

def build_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, data in zip_contents.items():
            zf.writestr(path, data)
    buf.seek(0)
    return buf.read()


# -------------------------------
# Excel-Export-Helfer (sichere Sheet-Namen)
# -------------------------------
INVALID_SHEET_CHARS = r'[:\\/?*\[\]]'
def sanitize_sheet_name(name: str, used: set) -> str:
    s = re.sub(INVALID_SHEET_CHARS, "_", str(name)).strip().strip("'")
    if not s:
        s = "Sheet"
    s = s[:31]  # Excel-Limit
    base = s
    i = 1
    while s in used:
        suffix = f"_{i}"
        s = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(s)
    return s

def to_bytes_excel_sanitized(sheets: dict) -> bytes:
    bio = io.BytesIO()
    used = set()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        for name, d in sheets.items():
            sname = sanitize_sheet_name(name, used)
            if isinstance(d, pd.DataFrame):
                d.to_excel(w, sheet_name=sname, index=False)
            elif isinstance(d, str):
                pd.DataFrame({"text": [d]}).to_excel(w, sheet_name=sname, index=False)
    bio.seek(0)
    return bio.getvalue()


# -------------------------------
# Plot-Helfer: Anzeige + SVG-Direktdownload + ZIP-Registrierung
# -------------------------------
def show_plot_with_svg(fig, rel_path: str, btn_label: str):
    """Zeigt den Plot in Streamlit, bietet direkten SVG-Download und legt die SVG auch in die ZIP."""
    st.pyplot(fig)
    svg_bytes = fig_to_svg(fig)
    key = "dl_" + re.sub(r"[^0-9a-zA-Z_]+", "_", rel_path)  # eindeutiger Key
    st.download_button(
        label=f"⬇️ {btn_label} (SVG)",
        data=svg_bytes,
        file_name=Path(rel_path).name,
        mime="image/svg+xml",
        key=key,
    )
    if opt_zip:
        zip_contents[rel_path] = svg_bytes


# -------------------------------------------------------
# 3️⃣ Analysen
# -------------------------------------------------------
st.header("3️⃣ Analysen")
anova_collect   = {}
means3d_collect = {}
heatmap_collect = {}

# Modelle & effektive Faktoren für What-If aufbewahren
models_by_y = {}
eff_factors_by_y = {}

with st.spinner("Berechne Analysen und erstelle Grafiken…"):
    for Y_safe in safe_results:
        Y_label = list(res_map.keys())[safe_results.index(Y_safe)]
        st.subheader(f"🎯 Ergebnis: **{Y_label}**")

        effective_factors = [F for F in safe_factors if df_work[F].nunique() >= 2]
        eff_factors_by_y[Y_safe] = list(effective_factors)

        if effective_factors:
            rhs     = " * ".join(effective_factors)
            formula = f"{Y_safe} ~ {rhs}"
        else:
            formula = f"{Y_safe} ~ 1"

        try:
            model = smf.ols(formula, data=df_work).fit()
            models_by_y[Y_safe] = model
        except Exception as e:
            st.error(f"❌ Modell für {Y_label} konnte nicht geschätzt werden: {e}")
            continue

        # ── ANOVA ──────────────────────────────────────────────────────────
        if opt_anova:
            if not effective_factors:
                st.info(f"ANOVA für {Y_label} übersprungen: keine Faktoren mit ≥2 Stufen.")
                dummy = pd.DataFrame({"Info": ["Keine gültigen Faktoren ≥2 Stufen"]})
                anova_collect[(Y_label, "TypeII")]  = dummy
                anova_collect[(Y_label, "TypeIII")] = dummy
            else:
                try:
                    a2 = anova_lm(model, typ=2)
                    a3 = anova_lm(model, typ=3)

                    def effect_sizes(anova_df):
                        tbl = anova_df.copy()
                        sst = tbl["sum_sq"].sum()
                        resid_idx = [i for i in tbl.index
                                     if "Residual" in str(i) or "resid" in str(i).lower()]
                        sse = tbl.loc[resid_idx[0], "sum_sq"] if resid_idx else np.nan
                        tbl["eta2"]         = tbl["sum_sq"] / sst
                        tbl["eta2_partial"] = tbl["sum_sq"] / (tbl["sum_sq"] + sse)
                        return tbl

                    es2 = effect_sizes(a2)
                    es3 = effect_sizes(a3)

                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("**ANOVA (Type II)**")
                        st.dataframe(a2)
                        st.markdown("**Effektstärken (Type II)**")
                        st.dataframe(es2[["sum_sq","df","F","PR(>F)","eta2","eta2_partial"]])
                    with colB:
                        st.markdown("**ANOVA (Type III)**")
                        st.dataframe(a3)
                        st.markdown("**Effektstärken (Type III)**")
                        st.dataframe(es3[["sum_sq","df","F","PR(>F)","eta2","eta2_partial"]])
                    
                    
                    # --- Download: ANOVA + Effektstärken (Excel) — direkt unter den Tabellen ---
                    try:
                        # Tabellen für Export vorbereiten
                        a2_out  = a2.reset_index().rename(columns={"index": "Term"})
                        a3_out  = a3.reset_index().rename(columns={"index": "Term"})
                        es2_out = es2.reset_index().rename(columns={"index": "Term"})
                        es3_out = es3.reset_index().rename(columns={"index": "Term"})
                        
                        # Einheitliche Reihenfolge der Effektstärken-Spalten (falls vorhanden)
                        cols_eff = [c for c in ["Term", "sum_sq", "df", "F", "PR(>F)", "eta2", "eta2_partial"] if c in es2_out.columns]
                        if cols_eff:
                            es2_out = es2_out[cols_eff]
                            es3_out = es3_out[cols_eff]
                            
                            # Excel-Sheets
                        sheets = {
                            "ANOVA_TypeII": a2_out,
                            "Effektstaerken_TypeII": es2_out,
                            "ANOVA_TypeIII": a3_out,
                            "Effektstaerken_TypeIII": es3_out,
                            }
                            
                        # Download-Button
                        st.download_button(
                            label=f"⬇️ ANOVA + Effektstärken für {Y_label} (Excel)",
                            data=to_bytes_excel_sanitized(sheets),
                            file_name=f"{Path(uploaded.name).stem}_{Y_label}_anova.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"btn_anova_{Y_safe}"
                            )
                    except Exception as e:
                        st.info(f"Download (ANOVA + Effektstärken) nicht verfügbar: {e}")
    
                    tbl2 = a2.reset_index().rename(columns={"index":"Term"})
                    tbl3 = a3.reset_index().rename(columns={"index":"Term"})
                    anova_collect[(Y_label, "TypeII")]  = tbl2
                    anova_collect[(Y_label, "TypeIII")] = tbl3
                    _register_df_csv(f"ANOVA/{Y_label}_TypeII.csv",  tbl2)
                    _register_df_csv(f"ANOVA/{Y_label}_TypeIII.csv", tbl3)
                except Exception as e:
                    st.warning(f"ANOVA für {Y_label} konnte nicht ausgeführt werden: {e}")
                    err = pd.DataFrame({"Fehler": [str(e)]})
                    anova_collect[(Y_label, "TypeII")]  = err
                    anova_collect[(Y_label, "TypeIII")] = err

        # ── Haupteffekte ────────────────────────────────────────────────────
        if opt_main and safe_factors:
            for F in safe_factors:
                origF = list(fac_map.keys())[safe_factors.index(F)]
                g = (df_work.groupby(F)[Y_safe]
                     .agg(mean="mean", std="std", n="count").reset_index())
                if g.empty:
                    continue
                g["se"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.errorbar(g[F].astype(str), g["mean"], yerr=g["se"], fmt="o-", capsize=5)
                ax.set_title(f"Haupteffekt: {origF} → {Y_label}")
                ax.set_xlabel(origF); ax.set_ylabel(Y_label)
                ax.grid(True, ls="--", alpha=0.4)
                plt.tight_layout()
                show_plot_with_svg(fig, f"Plots/{Y_label}/main_{Y_label}_{origF}.svg", f"SVG: Haupteffekt {origF}")
                plt.close(fig)

        # ── Interaktionen ───────────────────────────────────────────────────
        if opt_inter and len(safe_factors) >= 2:
            pairs = [(a, b) for i, a in enumerate(safe_factors) for b in safe_factors[i+1:]]
            for F1, F2 in pairs:
                origF1 = list(fac_map.keys())[safe_factors.index(F1)]
                origF2 = list(fac_map.keys())[safe_factors.index(F2)]
                fig, ax = plt.subplots(figsize=(6, 4))
                plotted = False
                for lvl, sub in df_work.groupby(F2):
                    g = sub.groupby(F1)[Y_safe].agg(mean="mean", std="std", n="count").reset_index()
                    if g.empty:
                        continue
                    g["se"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
                    ax.errorbar(g[F1].astype(str), g["mean"], yerr=g["se"],
                                fmt="o-", capsize=4, label=f"{origF2}={lvl}")
                    plotted = True
                if plotted:
                    ax.set_title(f"Interaktion: {origF1} × {origF2} → {Y_label}")
                    ax.set_xlabel(origF1); ax.set_ylabel(Y_label)
                    ax.legend(fontsize=8, ncol=2)
                    ax.grid(True, ls="--", alpha=0.4)
                    plt.tight_layout()
                    show_plot_with_svg(fig, f"Plots/{Y_label}/int_{Y_label}_{origF1}x{origF2}.svg",
                                       f"SVG: Interaktion {origF1}×{origF2}")
                plt.close(fig)

        # ── Boxplots ────────────────────────────────────────────────────────
        if opt_box and safe_factors:
            for F in safe_factors:
                origF = list(fac_map.keys())[safe_factors.index(F)]
                if df_work[F].dtype.name == "category" and df_work[F].nunique() >= 3:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x=F, y=Y_safe, data=df_work, ax=ax, showfliers=True)
                    ax.set_title(f"Boxplot: {origF} → {Y_label}")
                    ax.set_xlabel(origF); ax.set_ylabel(Y_label)
                    plt.tight_layout()
                    show_plot_with_svg(fig, f"Plots/{Y_label}/box_{Y_label}_{origF}.svg", f"SVG: Boxplot {origF}")
                    plt.close(fig)

        # ── Residuen ────────────────────────────────────────────────────────
        if opt_resid:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
            ax1.scatter(model.fittedvalues, model.resid, alpha=0.7)
            ax1.axhline(0, color="k", lw=1)
            ax1.set_xlabel("Fitted"); ax1.set_ylabel("Residuen")
            ax1.set_title("Residuen vs. Fitted")
            sm.qqplot(model.resid, line="45", fit=True, ax=ax2)
            ax2.set_title("Normal‑QQ")
            fig.suptitle(f"Residuen – {Y_label}")
            plt.tight_layout()
            show_plot_with_svg(fig, f"Plots/{Y_label}/residuals_{Y_label}.svg", "SVG: Residuen")
            plt.close(fig)

        # ── Daniel‑Plot ─────────────────────────────────────────────────────
        if opt_daniel and len(safe_factors) >= 2:
            two_level = [F for F in safe_factors
                         if df_work[F].dtype.name == "category" and df_work[F].nunique() == 2]
            if len(two_level) >= 2:
                df_d = df_work.copy()
                for F in two_level:
                    codes = df_d[F].cat.codes
                    df_d[F + "_pm1"] = codes.replace({0: -1, 1: 1, -1: np.nan}).astype(float)
                rhs_d = " * ".join([F + "_pm1" for F in two_level])
                try:
                    mod_d  = smf.ols(f"{Y_safe} ~ {rhs_d}", data=df_d).fit()
                    tvals  = mod_d.tvalues.drop(labels=["Intercept"], errors="ignore").abs().sort_values()
                    if len(tvals) > 0:
                        p  = (np.arange(1, len(tvals)+1) - 0.375) / (len(tvals) + 0.25)
                        hn = halfnorm.ppf(p)
                        fig, ax = plt.subplots(figsize=(6.5, 4.5))
                        ax.scatter(hn, tvals.values, color="steelblue", zorder=3)
                        coef = np.polyfit(hn, tvals.values, 1)
                        ax.plot(hn, np.poly1d(coef)(hn), color="k", lw=1.2)
                        for xi, yi, lab in zip(hn, tvals.values, tvals.index):
                            ax.annotate(lab, (xi, yi), xytext=(5, 3),
                                        textcoords="offset points", fontsize=8)
                        ax.set_title(f"Daniel (Half‑Normal) – {Y_label}")
                        ax.set_xlabel("Half‑Normal‑Quantile")
                        ax.set_ylabel("|t|-Werte")
                        plt.tight_layout()
                        show_plot_with_svg(fig, f"Plots/{Y_label}/daniel_{Y_label}.svg", "SVG: Daniel-Plot")
                        plt.close(fig)
                except Exception as e:
                    st.info(f"Daniel‑Plot für {Y_label} übersprungen: {e}")

        # ── 3D & Heatmap ───────────────────────────────────────────────────
        if opt_3d and len(safe_factors) >= 2 and f1_3d and f2_3d:
            origF1 = list(fac_map.keys())[safe_factors.index(f1_3d)]
            origF2 = list(fac_map.keys())[safe_factors.index(f2_3d)]

            means = df_work.groupby([f1_3d, f2_3d])[Y_safe].mean().reset_index()
            means_export = means.rename(columns={f1_3d: origF1, f2_3d: origF2, Y_safe: Y_label})
            means3d_collect[(Y_label, f"{origF1} x {origF2}")] = means_export.copy()
            _register_df_csv(f"3D/{Y_label}_{origF1}_x_{origF2}_means.csv", means_export)

            # Heatmap
#            if not means_export.empty:
#                try:
#                    pivot = means_export.pivot(index=origF2, columns=origF1, values=Y_label)
#                    heatmap_collect[(Y_label, f"{origF1} x {origF2}")] = pivot
##                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
 #                   sns.heatmap(pivot, annot=True, fmt=".3g", cmap="viridis", ax=ax,
  #                              linewidths=0.5, linecolor="grey")
##                    ax.set_title(f"2D-Heatmap der Mittelwerte: {origF1} × {origF2} → {Y_label}")
 #                   plt.tight_layout()
 #                   show_plot_with_svg(fig, f"Plots/{Y_label}/heatmap_{Y_label}_{origF1}_{origF2}.svg", "SVG: Heatmap")
 #                   plt.close(fig)
 #               except Exception as e:
 #                   st.warning(f"Heatmap für {Y_label} übersprungen: {e}")

            # 3D – Triangulation/Scatter
            if len(means) >= 3:
                try:
                    cat_x = means[f1_3d].astype("category")
                    cat_y = means[f2_3d].astype("category")
                    X_vals = cat_x.cat.codes.to_numpy(dtype=float)
                    Y_vals = cat_y.cat.codes.to_numpy(dtype=float)
                    Z_vals = means[Y_safe].to_numpy(dtype=float)

                    fig = plt.figure(figsize=(8, 6))
                    ax3d = fig.add_subplot(111, projection="3d")

                    try:
                        tri = mtri.Triangulation(X_vals, Y_vals)
                        ax3d.plot_trisurf(
                            X_vals, Y_vals, Z_vals,
                            triangles=tri.triangles, cmap="viridis",
                            edgecolor="none", alpha=0.9
                        )
                    except Exception:
                        pass

                    ax3d.scatter(X_vals, Y_vals, Z_vals, c=Z_vals, cmap="viridis", s=60,
                                 edgecolors="k", linewidths=0.4, zorder=5)

                    ax3d.set_xticks(np.arange(len(cat_x.cat.categories)))
                    ax3d.set_xticklabels(list(cat_x.cat.categories), rotation=30, ha="right", fontsize=8)
                    ax3d.set_yticks(np.arange(len(cat_y.cat.categories)))
                    ax3d.set_yticklabels(list(cat_y.cat.categories), rotation=30, ha="right", fontsize=8)
                    ax3d.set_xlabel(f"\n{origF1}", fontsize=9)
                    ax3d.set_ylabel(f"\n{origF2}", fontsize=9)
                    ax3d.set_zlabel(f"\n{Y_label}", fontsize=9)
                    ax3d.set_title(f"3D: {origF1} × {origF2} → {Y_label}", pad=10)
                    plt.tight_layout()
                    show_plot_with_svg(fig, f"Plots/{Y_label}/surface_{Y_label}_{origF1}_{origF2}.svg", "SVG: 3D-Plot")
                    plt.close(fig)
                except Exception as e:
                    st.error(f"3D‑Plot Fehler: {e}")


# -------------------------------
# PCA (über alle gewählten Ergebnisse; nur wenn ≥2)
# -------------------------------
if opt_pca and len(safe_results) >= 2:
    st.subheader("🧭 PCA – Scree, Loadings, Biplot")
    X = df_work[safe_results].dropna().to_numpy()
    if X.shape[0] >= 2:
        scaler    = StandardScaler()
        X_std     = scaler.fit_transform(X)
        pca       = PCA()
        scores    = pca.fit_transform(X_std)
        explained = pca.explained_variance_ratio_
        cum_exp   = np.cumsum(explained)

        # Scree
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(explained)+1), explained,  "o-", label="Varianzanteil")
        ax.plot(range(1, len(cum_exp)+1),    cum_exp,   "s--", label="kumuliert")
        ax.set_xticks(range(1, len(explained)+1))
        ax.set_xlabel("Komponente"); ax.set_ylabel("Varianzanteil")
        ax.set_title("Scree Plot"); ax.legend(); ax.grid(True)
        plt.tight_layout()
        show_plot_with_svg(fig, "Plots/PCA/scree.svg", "SVG: Scree")
        plt.close(fig)

        # Loadings
        res_labels = [list(res_map.keys())[safe_results.index(s)] for s in safe_results]
        loadings = pd.DataFrame(
            pca.components_.T,
            index=res_labels,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        _register_df_csv("PCA/loadings.csv", loadings.reset_index().rename(columns={"index":"Variable"}))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
        if X.shape[1] >= 2:
            for r in loadings.index:
                x, y = loadings.loc[r, "PC1"], loadings.loc[r, "PC2"]
                ax.arrow(0, 0, x, y, color="red", head_width=0.03)
                ax.text(x*1.1, y*1.1, r, fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Loadings Plot")
        ax.grid(True); plt.tight_layout()
        show_plot_with_svg(fig, "Plots/PCA/loadings.svg", "SVG: Loadings")
        plt.close(fig)

        # Biplot
        if X.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(scores[:, 0], scores[:, 1], alpha=0.6, color="blue")
            scale = 3
            for r in loadings.index:
                ax.arrow(0, 0, loadings.loc[r, "PC1"]*scale, loadings.loc[r, "PC2"]*scale,
                         color="red", head_width=0.1)
                ax.text(loadings.loc[r, "PC1"]*scale*1.15,
                        loadings.loc[r, "PC2"]*scale*1.15, r, color="red")
            pc1 = explained[0]*100 if len(explained) > 0 else 0
            pc2 = explained[1]*100 if len(explained) > 1 else 0
            ax.set_xlabel(f"PC1 ({pc1:.1f}%)"); ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
            ax.set_title("Biplot"); ax.grid(True); plt.tight_layout()
            show_plot_with_svg(fig, "Plots/PCA/biplot.svg", "SVG: Biplot")
            plt.close(fig)
    else:
        st.info("PCA: Zu wenige Zeilen nach NA‑Filter.")



# ============================================================
# What‑If – Kategoriale Faktoren filtern + numerisch interpolieren
# ============================================================
from scipy.interpolate import griddata

def render_whatif():
    """What‑If Block:
    - kategoriale Faktoren: Auswahl (kein Interpolieren)
    - numerische Faktoren: lineare Interpolation
    """

    if not opt_whatif or whatif_y is None:
        return

    # Zielvariable bestimmen
    try:
        y_col = whatif_y
        Y_label = list(res_map.keys())[safe_results.index(y_col)]
    except:
        st.error("❌ What‑If: Zielvariable konnte nicht identifiziert werden.")
        return

    st.subheader(f"Wert Interpolieren – {Y_label}")

    # --------------------------------------------------------
    # Faktoren trennen: numerisch vs. kategorial
    # --------------------------------------------------------
    numeric_factors = []
    categorical_factors = []

    for F in safe_factors:
        col = df_work[F]

        # Kann die Spalte numerisch werden?
        try:
            converted = pd.to_numeric(col, errors="coerce")
            if converted.notna().any():
                # Nur numerisch, wenn keine (!) Strings außer NaN übrig bleiben
                if col.dtype.kind in ("i", "f"):
                    numeric_factors.append(F)
                else:
                    # check: wenn ALLE Werte zu Zahlen werden → numerisch
                    if (converted.notna()).sum() == len(col):
                        numeric_factors.append(F)
                    else:
                        categorical_factors.append(F)
            else:
                categorical_factors.append(F)
        except:
            categorical_factors.append(F)

    # --------------------------------------------------------
    # Eingaben sammeln
    # --------------------------------------------------------
    selected_category_levels = {}
    numeric_inputs = {}

    st.markdown("### 🟦 Kategoriale Faktoren")
    if categorical_factors:
        for F in categorical_factors:
            origF = list(fac_map.keys())[safe_factors.index(F)]
            levels = sorted(df_work[F].dropna().unique())
            if len(levels) == 0:
                st.warning(f"⚠️ Faktor {origF} hat keine gültigen Stufen.")
                continue
            sel = st.selectbox(
                f"Stufe für **{origF}** wählen:",
                options=levels,
                key=f"cat_{F}"
            )
            selected_category_levels[F] = sel
    else:
        st.info("Keine kategorialen Faktoren vorhanden.")

    st.markdown("### 🟩 Numerische Faktoren")
    if numeric_factors:
        for F in numeric_factors:
            origF = list(fac_map.keys())[safe_factors.index(F)]
            mean_v = float(pd.to_numeric(df_work[F], errors="coerce").mean())
            numeric_inputs[F] = st.number_input(
                f"Wert für **{origF}** eingeben:",
                value=mean_v,
                key=f"num_{F}"
            )
    else:
        st.warning("❌ Keine numerischen Faktoren → Interpolation nicht möglich.")
        return

    # --------------------------------------------------------
    # Datensatz zuerst nach kategorialen Faktoren filtern
    # --------------------------------------------------------
    df_filtered = df_work.copy()
    for F, level in selected_category_levels.items():
        df_filtered = df_filtered[df_filtered[F] == level]

    if df_filtered.empty:
        st.error("❌ Keine Daten für die gewählten kategorialen Stufen vorhanden.")
        return

    # --------------------------------------------------------
    # Interpolation vorbereiten
    # --------------------------------------------------------
    try:
        X = df_filtered[numeric_factors].apply(pd.to_numeric, errors="coerce").to_numpy()
        y = pd.to_numeric(df_filtered[y_col], errors="coerce").to_numpy()
    except Exception as e:
        st.error(f"❌ Konnte Daten nicht vorbereiten: {e}")
        return

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    if len(X) < 3:
        st.error("❌ Für diese Materialkombination / Stufen zu wenige Daten für Interpolation.")
        return

    x_new = np.array([[numeric_inputs[F] for F in numeric_factors]])


    # --------------------------------------------------------
    # Interpolation (linear + nearest fallback)
    # --------------------------------------------------------
    try:
        y_pred = griddata(X, y, x_new, method="linear")
        if y_pred is None or np.isnan(y_pred).all():
            y_pred = griddata(X, y, x_new, method="nearest")
    except:
        try:
            y_pred = griddata(X, y, x_new, method="nearest")
        except Exception as e:
            st.error(f"❌ Interpolation fehlgeschlagen: {e}")
            return

    # --------------------------------------------------------
    # EXTRAPOLATION WARNUNG  ←  HIER kommt der Code hin
    # --------------------------------------------------------
    extrapolated = False
    eps = 1e-9  # Toleranz

    for F in numeric_factors:
        colvals = pd.to_numeric(df_filtered[F], errors="coerce").dropna().values
        min_v = np.min(colvals)
        max_v = np.max(colvals)
        user_v = numeric_inputs[F]

        if user_v < min_v - eps or user_v > max_v + eps:
            extrapolated = True
            st.warning(
                f"⚠️ **Extrapolation für Faktor {F}:** "
                f"Eingegebener Wert ({user_v}) liegt außerhalb des Messbereichs "
                f"({min_v} – {max_v}). Das Ergebnis kann ungenau sein."
            )

    # --------------------------------------------------------
    # Interpolationswert extrahieren
    # --------------------------------------------------------
    try:
        pred = float(np.asarray(y_pred).reshape(-1)[0])
    except:
        st.error("❌ Konnte Interpolationswert nicht extrahieren.")
        return

    # --------------------------------------------------------
    # Ausgabe
    # --------------------------------------------------------
    st.success(f"📌 **Vorhersage: {pred:.6g}**")

    df_out = pd.DataFrame({
        "Faktor": list(selected_category_levels.keys()) + numeric_factors,
        "Wert":   list(selected_category_levels.values()) +
                  [numeric_inputs[F] for F in numeric_factors]
    })
    df_out["Vorhersage"] = pred
    st.dataframe(df_out)

    if opt_zip:
        try:
            _register_df_csv(f"WhatIf/{Y_label}_interpolated.csv", df_out)
        except:
            pass


# Aufrufen
render_whatif()

# -------------------------------
# 📥 Downloads (Tabellen)
# -------------------------------
#st.markdown("### 📥 Downloads (Tabellen)")

#if anova_collect:
#    sheets = {}
#    for (ylab, kind), df_tbl in anova_collect.items():
#        sheets[f"{ylab}_{kind}"] = df_tbl  # enthält jetzt ANOVA + Effektstärken
#    st.download_button(
#        "ANOVA-Tabellen + Effektstärken als Excel",
#        data=to_bytes_excel_sanitized(sheets),
#        file_name=f"{Path(uploaded.name).stem}_anova.xlsx",
#        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#    )


#if means3d_collect:
#    sheets = {}
#    for (ylab, pairname), df_tbl in means3d_collect.items():
##        sheets[f"{ylab}_{pairname}__means"] = df_tbl
# ##       hpivot = heatmap_collect.get((ylab, pairname))
#        if hpivot is not None:
#            sheets[f"{ylab}_{pairname}__heatmap"] = hpivot.reset_index()
#    st.download_button(
#        "3D-Mittelwerte (und Heatmap) als Excel",
##        data=to_bytes_excel_sanitized(sheets),
 #       file_name=f"{Path(uploaded.name).stem}_3D_means.xlsx",
 #       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#    )


# --- ZIP NUR SVG + ANOVA + WHAT-IF ---
#if opt_zip:
#    readme = ["SVP ZIP Export", f"Quelle: {uploaded.name}", "", "Enthalten:"]
#    for p in sorted(zip_contents.keys()):
##        readme.append(f"  {p}")
#    zip_contents["README.txt"] = "\n".join(readme).encode("utf-8")#
#
#    b = io.BytesIO()
#    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
#        for p, d in zip_contents.items():
#            z.writestr(p, d)
#    b.seek(0)

#    st.download_button(
 #       "⬇️ ZIP herunterladen (SVG + ANOVA + What‑If)",
  #      b.getvalue(),
   #     file_name=f"SVP_Export_{Path(uploaded.name).stem}.zip",
    #    mime="application/zip"
    #)