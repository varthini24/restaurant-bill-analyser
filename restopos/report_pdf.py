"""Builds the downloadable PDF report (ReportLab) from the analysis results."""
import io
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

FIG_TITLES = [
    ("fig01_missing", "Figure 1: Missing values introduced per column (demonstration copy)"),
    ("fig02_hist", "Figure 2: Histograms of bill total, prep time and rating (dashed = mean)"),
    ("fig03_box", "Figure 3: Box plots and outliers (1.5×IQR rule)"),
    ("fig04_type", "Figure 4: Mean bill, prep time and rating by order type"),
    ("fig05_weekend_pay", "Figure 5: Mean bill by weekday/weekend and by payment method"),
    ("fig06_category", "Figure 6: Revenue by menu category"),
    ("fig07_heatmap", "Figure 7: Correlation heatmap"),
    ("fig08_reg_total", "Figure 8: Simple regression, items vs bill total"),
    ("fig09_reg_prep", "Figure 9: Simple regression, items vs prep time"),
    ("fig10_resid", "Figure 10: Residuals vs fitted and residual histogram"),
    ("fig11_actual_pred", "Figure 11: Actual vs predicted prep time (test set)"),
    ("fig12_coefs", "Figure 12: Multiple regression coefficients"),
]


def _tbl(rows, widths=None, head=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if head else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]
    if head: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t


def build_pdf(R, extra_figs, period):
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor("#1f4e79"), fontSize=16, spaceBefore=10)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#1f4e79"), fontSize=12)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13); CAP = ParagraphStyle("C", parent=B, fontSize=8.5, textColor=colors.HexColor("#475569"), alignment=1)
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="Restaurant Sales Analysis Report")
    d, m1, m2, m4 = R["d"], R["m1"], R["m2"], R["m4"]
    S = []
    f = lambda x, n=2: f"{x:,.{n}f}"
    S += [Paragraph("Restaurant Sales &amp; Order Analysis Report", ParagraphStyle("T", parent=ss["Title"], textColor=colors.HexColor("#1f4e79"))),
          Paragraph(f"Period: {period} &nbsp;|&nbsp; Orders analysed: {len(d):,} &nbsp;|&nbsp; Tools: NumPy, Pandas, Matplotlib, SciPy, scikit-learn", B), Spacer(1, 8)]
    kp = [["Total revenue (net)", "Orders", "Avg bill", "Gross profit", "Orders rated >=4"],
          [f"Rs {d.net_sales.sum():,.0f}", f"{len(d):,}", f"Rs {d.total.mean():,.0f}", f"Rs {d.profit.sum():,.0f}", f"{R['pass_rate']:.1f}%"]]
    S += [_tbl(kp), Spacer(1, 8), Paragraph("1. Summary of findings", H1)]
    top = R["by_cat"].index[0]
    for t in [
        f"The dataset has {len(d):,} orders with {R['dups']} duplicate order ids. Mean bill is Rs {d.total.mean():,.0f} (median Rs {d.total.median():,.0f}); mean preparation time is {d.prep_min.mean():.1f} min and mean rating {d.rating.mean():.2f}.",
        f"{top} is the highest-revenue category (Rs {R['by_cat'].revenue.iloc[0]:,.0f}). Dine-in bills average Rs {R['by_type'].loc['Dine in','total']:,.0f}, about {R['by_type'].loc['Dine in','total']/R['by_type'].loc['Take away','total']:.1f}× take-away bills.",
        f"Items per order explains {m1['r2']*100:.1f}% of bill-total variation (simple regression, slope Rs {m1['beta'][1]:.1f} per item) and {m2['r2']*100:.1f}% of prep-time variation.",
        f"The multiple regression for prep time (items, party size, weekend, peak hour, delivery) reaches R<super>2</super> = {m4['r2']:.3f} (adjusted {m4['adj']:.3f}); on the unseen 20% test set R<super>2</super> = {R['test_multi']['r2']:.3f} and MAE = {R['test_multi']['mae']:.2f} min, versus MAE {R['test_simple']['mae']:.2f} min for the simple model.",
        f"Peak hours (1 pm and 8 pm) add about {m4['beta'][4]:.1f} min and delivery about {m4['beta'][5]:.1f} min to preparation time at the same order size. All results describe association, not causation."]:
        S.append(Paragraph("• " + t, B))

    S += [Paragraph("2. Descriptive statistics", H1)]
    ds = R["describe"]; rows = [["stat"] + list(ds.columns)] + [[i] + [f(v) for v in r] for i, r in ds.iterrows()]
    S += [_tbl(rows), Spacer(1, 6), Paragraph("Group means by order type", H2)]
    bt = R["by_type"]; S.append(_tbl([["Order type"] + list(bt.columns)] + [[i] + [f(v) for v in r] for i, r in bt.iterrows()]))
    S += [Paragraph("Revenue by category", H2)]
    bc = R["by_cat"]; S.append(_tbl([["Category", "Revenue (Rs)", "Units"]] + [[i, f(r.revenue, 0), f(r.units, 0)] for i, r in bc.iterrows()]))

    S += [Paragraph("3. Missing-data handling", H1),
          Paragraph(f"Real missing values found in the stored orders: {sum(R['real_missing'].values())} (bills added manually have no prep time or rating until recorded; these are median-imputed). "
                    f"To demonstrate the method, {sum(R['missing_intro'].values())} further values were removed at random (seed 42) from a working copy, then filled with the median (numeric) and mode ('{R['mode_payment']}') for the categorical column. "
                    f"Missing after treatment: {R['missing_after']}.", B),
          _tbl([["Column", "Mean (orig)", "Mean (imputed)", "Std (orig)", "Std (imputed)"]] + [[c, f(a), f(b), f(c1), f(c2)] for c, a, b, c1, c2 in R["impute_cmp"]])]

    S += [Paragraph("4. Regression analysis", H1), Paragraph("Model 1 (simple): bill total ~ items; Model 2 (simple): prep time ~ items; Model 3 (multiple): prep time ~ items + party size + weekend + peak hour + delivery. Estimated by ordinary least squares in NumPy.", B)]
    def coef(m): return _tbl([["Predictor", "Coef", "Std err", "t", "p-value", "95% CI"]] + [[n, f(b, 3), f(s, 3), f(t, 2), "<0.001" if p < 0.001 else f(p, 3), f"{lo:.2f} to {hi:.2f}"] for n, b, s, t, p, (lo, hi) in zip(m["names"], m["beta"], m["se"], m["t"], m["p"], m["ci"])])
    for nm, m in [("Model 1: bill total ~ items", m1), ("Model 2: prep time ~ items", m2), ("Model 3: prep time ~ 5 predictors", m4)]:
        S += [Paragraph(nm, H2), coef(m), Paragraph(f"R<super>2</super> = {m['r2']:.3f}, adjusted R<super>2</super> = {m['adj']:.3f}, F = {m['F']:.1f} (p {'< 0.001' if m['fp'] < .001 else f'= {m['fp']:.3f}'}), RMSE = {m['rmse']:.2f}, n = {m['n']}", B)]
    S += [Paragraph("5. Model evaluation and diagnostics", H1),
          _tbl([["Model (prep time)", "R-sq", "MAE", "RMSE"], ["Simple (items only)", f(R["test_simple"]["r2"], 3), f(R["test_simple"]["mae"]), f(R["test_simple"]["rmse"])], ["Multiple (5 predictors)", f(R["test_multi"]["r2"], 3), f(R["test_multi"]["mae"]), f(R["test_multi"]["rmse"])]]),
          Paragraph(f"Residuals: mean {R['resid']['mean']:.2f}, std {R['resid']['std']:.2f}, skew {R['resid']['skew']:.3f}; Shapiro-Wilk p = {R['resid']['shapiro_p']:.3f}; Durbin-Watson = {m4['dw']:.2f}. Variance inflation factors: " + ", ".join(f"{n} {v:.2f}" for n, v in R["vif"]) + ". A VIF above about 10 would signal multicollinearity.", B)]

    S += [PageBreak(), Paragraph("6. Figures", H1)]
    for key, cap in FIG_TITLES:
        im = Image(io.BytesIO(R["figs"][key])); r = min(15.5 * cm / im.imageWidth, 9 * cm / im.imageHeight); im.drawWidth, im.drawHeight = im.imageWidth * r, im.imageHeight * r
        S.append(KeepTogether([im, Paragraph(cap, CAP), Spacer(1, 8)]))
    for key, cap in [("time_of_day", "Figure 13: Net sales by time of day, this week vs last week"), ("day_of_week", "Figure 14: Net sales by day of week, this week vs last week")]:
        if key in extra_figs:
            im = Image(io.BytesIO(extra_figs[key])); r = min(15.5 * cm / im.imageWidth, 9 * cm / im.imageHeight); im.drawWidth, im.drawHeight = im.imageWidth * r, im.imageHeight * r
            S.append(KeepTogether([im, Paragraph(cap, CAP), Spacer(1, 8)]))
    S += [Paragraph("7. Limitations", H1), Paragraph("Order history in this project is simulated with NumPy until real bills are added through the billing page. Results describe association, not causation; only linear models were fitted; the 'rating >= 4' threshold is an assumption.", B)]
    doc.build(S); return buf.getvalue()
