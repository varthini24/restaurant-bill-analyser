"""Analysis engine (NumPy + Pandas + Matplotlib + SciPy + scikit-learn).
Follows the same pipeline as the DSA report: manipulation -> missing data -> EDA -> regression -> diagnostics."""
import io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import data_gen as dg

C1, C2, C3, GR = "#1f4e79", "#c0504d", "#4f8a4f", "#7f7f7f"
CYAN, ORANGE, GREEN = "#38bdc9", "#ff8a00", "#a3d977"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
_cache = {}


def png(fig, dpi=130, transparent=False):
    b = io.BytesIO(); fig.tight_layout(); fig.savefig(b, format="png", dpi=dpi, transparent=transparent); plt.close(fig)
    return b.getvalue()


# ------------------------------------------------------------------ OLS in NumPy (replaces statsmodels)
def ols(X, y, names):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)]); y = np.asarray(y, float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fit = X @ beta; res = y - fit
    rss = float(res @ res); tss = float(((y - y.mean()) ** 2).sum())
    s2 = rss / (n - k); cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov)); t = beta / se; p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - rss / tss; adj = 1 - (1 - r2) * (n - 1) / (n - k)
    F = ((tss - rss) / (k - 1)) / s2; fp = stats.f.sf(F, k - 1, n - k)
    tc = stats.t.ppf(0.975, n - k)
    dw = float(np.sum(np.diff(res) ** 2) / rss)
    return dict(names=["Intercept"] + list(names), beta=beta, se=se, t=t, p=p, r2=r2, adj=adj, F=F, fp=fp,
                rmse=float(np.sqrt(s2)), n=n, fit=fit, res=res, dw=dw,
                ci=[(b - tc * s, b + tc * s) for b, s in zip(beta, se)])


def vif(df, cols):
    out = []
    for c in cols:
        others = [x for x in cols if x != c]
        m = ols(df[others].values, df[c].values, others)
        out.append((c, 1 / (1 - m["r2"]) if m["r2"] < 1 else np.inf))
    return out


# ------------------------------------------------------------------ pipeline
def prepare(start=None, end=None):
    orders, lines = dg.load_orders()
    if start: orders = orders[orders.datetime >= pd.Timestamp(start)]
    if end: orders = orders[orders.datetime < pd.Timestamp(end) + pd.Timedelta(days=1)]
    lines = lines[lines.order_id.isin(orders.order_id)]
    d = orders.copy()
    d["date"] = d.datetime.dt.normalize(); d["hour"] = d.datetime.dt.hour; d["dow"] = d.datetime.dt.dayofweek
    d["is_weekend"] = (d.dow >= 5).astype(int)
    d["net_sales"] = d.subtotal - d.discount
    d["profit"] = d.net_sales - d.cost
    d["bill_size"] = np.where(d.total >= d.total.median(), "High", "Low")          # np.where derived column
    d["good_rating"] = pd.Series(np.where(d.rating >= 4, "Yes", "No"), index=d.index).where(d.rating.notna())
    return d, lines


def analyse(start=None, end=None):
    key = (os.path.getmtime(dg.ORDERS_CSV) if os.path.exists(dg.ORDERS_CSV) else 0, start, end)
    if key in _cache: return _cache[key]
    d, lines = prepare(start, end)
    R, F = {}, {}
    num = ["total", "prep_min", "rating"]

    # ---- 1. inspection + real missing data
    R["shape"] = d.shape; R["dups"] = int(d.duplicated("order_id").sum())
    R["real_missing"] = d[["subtotal", "prep_min", "rating", "payment"]].isna().sum().to_dict()
    clean = d.copy()
    for c in ["prep_min", "rating"]:
        clean[c] = clean[c].fillna(clean[c].median())
    clean["payment"] = clean["payment"].fillna(clean["payment"].mode()[0])
    clean["good_rating"] = np.where(clean.rating >= 4, "Yes", "No")

    # ---- 2. missing-data demonstration (MCAR, seed 42) on a working copy, exactly like the report
    rng = np.random.default_rng(42)
    dfm = clean.copy(); n = len(dfm)
    plan = [("total", max(1, int(n * 0.04))), ("prep_min", max(1, int(n * 0.035))), ("rating", max(1, int(n * 0.03))), ("payment", max(1, int(n * 0.025)))]
    for col, k in plan:
        dfm.loc[rng.choice(dfm.index, k, replace=False), col] = np.nan
    R["missing_intro"] = dfm[[c for c, _ in plan]].isna().sum().to_dict()
    imp = dfm.copy()
    for c in ["total", "prep_min", "rating"]: imp[c] = imp[c].fillna(imp[c].median())
    imp["payment"] = imp["payment"].fillna(imp["payment"].mode()[0])
    R["mode_payment"] = imp["payment"].mode()[0]
    R["missing_after"] = int(imp[[c for c, _ in plan]].isna().sum().sum())
    R["impute_cmp"] = [(c, clean[c].mean(), imp[c].mean(), clean[c].std(), imp[c].std()) for c in num]
    d = clean  # modelling uses the complete cleaned data

    # ---- 3. descriptive statistics + group-wise aggregation
    R["describe"] = d[["total", "items_count", "prep_min", "rating", "profit"]].describe().round(2)
    R["skew"] = d[num].skew().round(3).to_dict()
    R["by_type"] = d.groupby("order_type")[num].mean().round(2)
    R["by_pay"] = d.groupby("payment")[num].mean().round(2)
    R["by_weekend"] = d.groupby("is_weekend")[num].mean().round(2).rename(index={0: "Weekday", 1: "Weekend"})
    cat = lines.assign(rev=lines.qty * lines.price).groupby("category").agg(revenue=("rev", "sum"), units=("qty", "sum")).sort_values("revenue", ascending=False)
    R["by_cat"] = cat.round(0)
    R["by_item"] = lines.assign(rev=lines.qty * lines.price).groupby("item_name").agg(units=("qty", "sum"), revenue=("rev", "sum")).sort_values("revenue", ascending=False)
    R["pass_rate"] = float((d.good_rating == "Yes").mean() * 100)
    R["corr"] = d[["total", "items_count", "prep_min", "rating"]].corr().round(3)
    out = {c: int(((d[c] < d[c].quantile(.25) - 1.5 * (d[c].quantile(.75) - d[c].quantile(.25))) | (d[c] > d[c].quantile(.75) + 1.5 * (d[c].quantile(.75) - d[c].quantile(.25)))).sum()) for c in num}
    R["outliers"] = out

    # ---- 4. regression
    feats = ["items_count", "party_size", "is_weekend", "is_peak", "delivery"]
    d["is_peak"] = d.hour.isin([13, 20]).astype(int); d["delivery"] = (d.order_type == "Delivery").astype(int)
    R["m1"] = ols(d[["items_count"]], d["total"], ["items_count"])          # simple: bill ~ items
    R["m2"] = ols(d[["items_count"]], d["prep_min"], ["items_count"])       # simple: prep time ~ items
    R["m4"] = ols(d[feats], d["prep_min"], feats)                           # multiple: prep time ~ 5 predictors
    R["feats"] = feats; R["vif"] = vif(d, feats)
    Xtr, Xte, ytr, yte = train_test_split(d[feats], d["prep_min"], test_size=0.2, random_state=42)
    lr = LinearRegression().fit(Xtr, ytr); pred = lr.predict(Xte)
    X1tr, X1te, y1tr, y1te = train_test_split(d[["items_count"]], d["prep_min"], test_size=0.2, random_state=42)
    p1 = LinearRegression().fit(X1tr, y1tr).predict(X1te)
    mk = lambda y, p: dict(r2=r2_score(y, p), mae=mean_absolute_error(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))))
    R["test_multi"], R["test_simple"] = mk(yte, pred), mk(y1te, p1); R["ntr"], R["nte"] = len(Xtr), len(Xte)
    res = R["m4"]["res"]
    R["resid"] = dict(mean=float(res.mean()), std=float(res.std()), skew=float(stats.skew(res)), shapiro_p=float(stats.shapiro(res).pvalue))

    # ---- 5. figures (12 report figures)
    def fig_(w, h): return plt.subplots(figsize=(w, h))
    fig, ax = fig_(6.2, 3.6); mi = pd.Series(R["missing_intro"]); ax.bar(mi.index, mi.values, color=C2)
    for i, v in enumerate(mi.values): ax.text(i, v + .5, str(v), ha="center")
    ax.set_ylabel("Missing values"); ax.set_title("Missing Values Introduced per Column"); F["fig01_missing"] = png(fig)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, c, col, t in zip(axs, num, [C1, C2, C3], ["Bill Total (₹)", "Prep Time (min)", "Customer Rating"]):
        ax.hist(d[c], bins=20, color=col, edgecolor="white"); ax.axvline(d[c].mean(), color="k", ls="--", lw=1); ax.set_title(t)
    axs[0].set_ylabel("Orders"); F["fig02_hist"] = png(fig)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, c, col, t in zip(axs, num, [C1, C2, C3], ["Bill Total (₹)", "Prep Time (min)", "Rating"]):
        bp = ax.boxplot(d[c], patch_artist=True, tick_labels=[t]); bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(.6)
    axs[0].set_ylabel("Value"); fig.suptitle("Distribution and Outliers (1.5×IQR rule)"); F["fig03_box"] = png(fig)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4)); bt = R["by_type"]
    for ax, c, t in zip(axs, num, ["Mean Bill (₹)", "Mean Prep (min)", "Mean Rating"]):
        ax.bar(bt.index, bt[c], color=[C1, C2, C3]); ax.set_title(t)
        for i, v in enumerate(bt[c]): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.suptitle("Mean Values by Order Type"); F["fig04_type"] = png(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bw = R["by_weekend"]["total"]; axs[0].bar(bw.index, bw.values, color=[C3, C1]); axs[0].set_title("Weekday vs Weekend: Mean Bill (₹)")
    bp_ = R["by_pay"]["total"]; axs[1].bar(bp_.index, bp_.values, color=[C1, C2, C3, GR][:len(bp_)]); axs[1].set_title("Payment Method: Mean Bill (₹)")
    for ax, s in zip(axs, [bw, bp_]):
        for i, v in enumerate(s.values): ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    F["fig05_weekend_pay"] = png(fig)

    fig, ax = fig_(7, 3.8); cs = R["by_cat"].revenue[::-1]
    ax.barh(cs.index, cs.values, color=C1)
    for i, v in enumerate(cs.values): ax.text(v, i, f" ₹{v:,.0f}", va="center")
    ax.set_xlim(0, cs.max() * 1.18); ax.set_xlabel("Revenue (₹)"); ax.set_title("Revenue by Menu Category"); F["fig06_category"] = png(fig)

    fig, ax = fig_(4.8, 4); cr = R["corr"]; im = ax.imshow(cr.values, cmap="Blues", vmin=-.3, vmax=1)
    lab = ["Total", "Items", "Prep", "Rating"]; ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{cr.values[i, j]:.2f}", ha="center", va="center", color="white" if cr.values[i, j] > .6 else "black")
    fig.colorbar(im, fraction=.046); ax.set_title("Correlation Heatmap"); F["fig07_heatmap"] = png(fig)

    def scat(x, y, m, xl, yl, title, col, name):
        fig, ax = fig_(6.2, 4.2); ax.scatter(d[x], d[y], s=14, alpha=.45, color=col)
        xs = np.linspace(d[x].min(), d[x].max(), 100)
        ax.plot(xs, m["beta"][0] + m["beta"][1] * xs, color=C2, lw=2, label=f"y = {m['beta'][0]:.2f} + {m['beta'][1]:.2f}x  (R² = {m['r2']:.3f})")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title); ax.legend(); F[name] = png(fig)
    scat("items_count", "total", R["m1"], "Items per order", "Bill total (₹)", "Simple Regression: Items vs Bill Total", C1, "fig08_reg_total")
    scat("items_count", "prep_min", R["m2"], "Items per order", "Prep time (min)", "Simple Regression: Items vs Prep Time", C3, "fig09_reg_prep")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].scatter(R["m4"]["fit"], res, s=12, alpha=.45, color=C1); axs[0].axhline(0, color=C2)
    axs[0].set(xlabel="Fitted prep time (min)", ylabel="Residuals (min)", title="Residuals vs Fitted")
    axs[1].hist(res, bins=25, color=C3, edgecolor="white"); axs[1].set(xlabel="Residual (min)", title="Residual Distribution"); F["fig10_resid"] = png(fig)

    fig, ax = fig_(5.4, 4.6); ax.scatter(yte, pred, s=18, alpha=.55, color=C1)
    lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]; ax.plot(lim, lim, color=C2, ls="--", label="Perfect prediction")
    ax.set(xlabel="Actual prep time (min)", ylabel="Predicted prep time (min)", title="Actual vs Predicted (Test Set)"); ax.legend(); F["fig11_actual_pred"] = png(fig)

    fig, ax = fig_(6.2, 3.6); nm = ["Items", "Party size", "Weekend", "Peak hour", "Delivery"]; vals = R["m4"]["beta"][1:]
    ax.barh(nm, vals, color=[C1 if v >= 0 else C2 for v in vals]); ax.axvline(0, color="k", lw=.8)
    for i, v in enumerate(vals): ax.text(v, i, f" {v:.1f} ", va="center", ha="left" if v >= 0 else "right")
    ax.set_xlabel("Coefficient (minutes of prep time)"); ax.set_title("Multiple Regression Coefficients"); F["fig12_coefs"] = png(fig)

    R["d"] = d; R["lines"] = lines; R["figs"] = F
    _cache.clear(); _cache[key] = R
    return R


# ------------------------------------------------------------------ dashboards / sales charts
def split_weeks(d):
    end = d.date.max(); this = d[d.date > end - pd.Timedelta(days=7)]; last = d[(d.date <= end - pd.Timedelta(days=7)) & (d.date > end - pd.Timedelta(days=14))]
    return this, last, end


def sales_charts(d, lines):
    this, last, end = split_weeks(d)
    fmt = lambda ax: (ax.yaxis.set_major_formatter(lambda v, _: f"₹{v:,.0f}"), ax.grid(axis="y", color="#e5e7eb", lw=.8), ax.set_axisbelow(True), [s.set_visible(False) for s in ax.spines.values()])
    lab = lambda a, b: (f"{(end - pd.Timedelta(days=6)).strftime('%b %d')} - {end.strftime('%b %d')}", f"{(end - pd.Timedelta(days=13)).strftime('%b %d')} - {(end - pd.Timedelta(days=7)).strftime('%b %d')}")
    L1, L2 = lab(0, 0); out = {}
    h = np.arange(10, 23); w = .4
    fig, ax = plt.subplots(figsize=(8.4, 4))
    ax.bar(h - w / 2, this.groupby("hour").net_sales.sum().reindex(h, fill_value=0), w, color=CYAN, label=L1)
    ax.bar(h + w / 2, last.groupby("hour").net_sales.sum().reindex(h, fill_value=0), w, color=ORANGE, label=L2)
    ax.set_xticks(h); ax.set_xticklabels([f"{x % 12 or 12}{'a' if x < 12 else 'p'}" for x in h]); fmt(ax); out["time_of_day"] = png(fig)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; x = np.arange(7)
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.bar(x - w / 2, this.groupby("dow").net_sales.sum().reindex(range(7), fill_value=0), w, color=CYAN, label=L1)
    ax.bar(x + w / 2, last.groupby("dow").net_sales.sum().reindex(range(7), fill_value=0), w, color=ORANGE, label=L2)
    ax.set_xticks(x); ax.set_xticklabels(days); fmt(ax); out["day_of_week"] = png(fig)
    return out


def dark_charts(d, lines):
    this, last, end = split_weeks(d); out = {}
    bg, fg, grid = "none", "#cbd5e1", "#334155"
    def style(ax):
        ax.set_facecolor("none"); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(colors=fg, length=0); ax.grid(axis="y", color=grid, lw=.6); ax.set_axisbelow(True)
    daily = d.groupby("date").net_sales.sum().tail(30)
    fig, ax = plt.subplots(figsize=(4.6, 1.5)); ax.bar(range(len(daily)), daily.values, color=GREEN, width=.55, alpha=.9)
    ax.axis("off"); out["spark"] = png(fig, transparent=True)
    g = d.groupby("date").agg(rev=("net_sales", "sum"), cost=("cost", "sum")).tail(7)
    fig, ax = plt.subplots(figsize=(5.4, 3.0)); x = np.arange(len(g)); w = .36
    ax.bar(x - w / 2, g.rev, w, color=GREEN, label="Revenue"); ax.bar(x + w / 2, g.cost, w, color="#475569", label="Expenses")
    ax.set_xticks(x); ax.set_xticklabels([t.strftime("%a") for t in g.index]); ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    style(ax); ax.legend(frameon=False, labelcolor=fg, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(.5, -.12)); out["profit"] = png(fig, transparent=True)
    top = lines[lines.order_id.isin(this.order_id)].groupby("item_name").qty.sum().sort_values().tail(6)
    fig, ax = plt.subplots(figsize=(4.4, 2.6)); ax.barh(top.index, top.values, color=GREEN, height=.55)
    style(ax); ax.grid(axis="x", color=grid, lw=.6); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", colors="#e2e8f0", labelsize=8)
    for i, v in enumerate(top.values): ax.text(v, i, f" {v}", va="center", color=fg, fontsize=8)
    out["top_items"] = png(fig, transparent=True)
    return out
