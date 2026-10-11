import io, os, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, abort, flash
from PIL import Image, ImageOps
import data_gen as dg
import analysis as an
from report_pdf import build_pdf, FIG_TITLES

app = Flask(__name__)
app.secret_key = "resto-pos-dev"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
dg.ensure_menu(); dg.load_orders()


@app.template_filter("inr")
def inr(v, d=0):
    return f"₹{v:,.{d}f}"


@app.context_processor
def inject():
    return dict(image_url=dg.image_url, CATEGORIES=dg.CATEGORIES, EMOJI=dg.CAT_EMOJI, nav=request.endpoint)


def rng_args():
    s, e = request.args.get("start") or None, request.args.get("end") or None
    return s, e


# ------------------------------------------------------------------ Dashboard (image 3)
@app.route("/")
def dashboard():
    d, lines = an.prepare()
    this, last, end = an.split_weeks(d)
    pct = lambda a, b: (a / b - 1) * 100 if b else 0
    weeks = max(1, (d.date.max() - d.date.min()).days / 7)
    rev_this, rev_last = this.net_sales.sum(), last.net_sales.sum()
    prof_this = this.profit.sum()
    rated = d.rating.dropna(); good = (rated >= 4).mean() * 100
    # busiest slot this week vs last week
    h = this.groupby("hour").size(); busiest = int(h.idxmax()); last_same = (last.hour == busiest).sum()
    slot_change = pct(h.max(), last_same)
    fmt_h = lambda x: f"{x % 12 or 12} {'AM' if x < 12 else 'PM'}"
    # trending item
    li = lines.merge(d[["order_id", "date"]], on="order_id")
    tw = li[li.date > end - pd.Timedelta(days=7)].groupby("item_name").qty.sum()
    lw = li[(li.date <= end - pd.Timedelta(days=7)) & (li.date > end - pd.Timedelta(days=14))].groupby("item_name").qty.sum()
    cmp_ = pd.concat([tw, lw], axis=1, keys=["this", "last"]).fillna(0)
    cmp_["growth"] = (cmp_["this"] - cmp_["last"]) / cmp_["last"].replace(0, np.nan) * 100
    best = cmp_.sort_values("this", ascending=False).head(8).sort_values("growth", ascending=False).iloc[0]
    best_name = best.name
    menu = dg.load_menu().set_index("name")
    charts = an.dark_charts(d, lines)
    import base64
    b64 = {k: base64.b64encode(v).decode() for k, v in charts.items()}
    recent = d.sort_values("datetime", ascending=False).head(12)
    return render_template("dashboard.html", rev=rev_this, rev_pct=pct(rev_this, rev_last), avg_week=d.net_sales.sum() / weeks, profit=prof_this,
                           avg_profit=d.profit.sum() / weeks, good=good, busiest=f"{fmt_h(busiest)} - {fmt_h(busiest + 1)}", slot_change=slot_change,
                           best_name=best_name, best_img=menu.loc[best_name, "image"], best_this=int(best["this"]), best_last=int(best["last"]),
                           best_growth=0 if pd.isna(best.growth) else best.growth, charts=b64, recent=recent, orders_n=len(d), end=end,
                           top_cat=lines.assign(r=lines.qty * lines.price).groupby("category").r.sum().idxmax())


# ------------------------------------------------------------------ POS menu (image 1)
@app.route("/menu")
def menu():
    m = dg.load_menu(); m = m[m.status == "Active"]
    return render_template("menu.html", items=m.to_dict("records"), cats=[c for c in dg.CATEGORIES if c in set(m.category)])


@app.route("/api/bill", methods=["POST"])
def api_bill():
    j = request.get_json(force=True)
    try:
        row = dg.add_bill([(i["id"], i["qty"]) for i in j["items"]], j.get("order_type", "Dine in"), j.get("payment", "Cash"), float(j.get("discount_pct", 0)),
                          party_size=int(j.get("party_size", 1)))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, order_id=row["order_id"], total=row["total"], subtotal=row["subtotal"], tax=row["tax"], discount=row["discount"], datetime=row["datetime"])


# ------------------------------------------------------------------ Item manager (image 2)
def _save_upload(file, item_id):
    if not file or not file.filename: return None
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"): raise ValueError("Please upload a PNG, JPG or WEBP image")
    img = ImageOps.exif_transpose(Image.open(file.stream)).convert("RGB")
    img = ImageOps.fit(img, (960, 660), Image.LANCZOS)  # crop-to-fit keeps every photo sharp and the same shape
    fn = f"item-{item_id}.jpg"; os.makedirs(os.path.join(dg.BASE, "static", "uploads"), exist_ok=True)
    img.save(os.path.join(dg.BASE, "static", "uploads", fn), "JPEG", quality=92)
    return fn


@app.route("/items")
def items():
    return render_template("items.html", items=dg.load_menu().to_dict("records"))


@app.route("/items/add", methods=["POST"])
def items_add():
    m = dg.load_menu(); nid = int(m.id.max()) + 1
    try:
        name = request.form["name"].strip(); price = float(request.form["price"])
        if not name or price <= 0: raise ValueError("Name and a positive price are required")
        fn = _save_upload(request.files.get("image"), nid)
    except Exception as e:
        flash(str(e), "err"); return redirect(url_for("items"))
    cat = request.form.get("category", "Pizza")
    if not fn:
        fn = dg.slugify(name) + ".jpg"
        if cat in dg.PAINT: dg.make_image(name, cat, os.path.join(dg.IMG, fn))
    row = dict(id=nid, name=name, category=cat, price=price, cost=round(price * 0.38, 2), status=request.form.get("status", "Active"), image=fn, description=request.form.get("description", ""))
    dg.save_menu(pd.concat([m, pd.DataFrame([row])])); flash(f"Added {name}", "ok")
    return redirect(url_for("items"))


@app.route("/items/<int:i>/edit", methods=["POST"])
def items_edit(i):
    m = dg.load_menu()
    if i not in set(m.id): abort(404)
    try:
        fn = _save_upload(request.files.get("image"), i)
        price = float(request.form["price"])
    except Exception as e:
        flash(str(e), "err"); return redirect(url_for("items"))
    for k in ("name", "category", "status", "description"):
        if k in request.form: m.loc[m.id == i, k] = request.form[k]
    m.loc[m.id == i, "price"] = price
    if fn: m.loc[m.id == i, "image"] = fn
    dg.save_menu(m); flash("Item updated", "ok"); return redirect(url_for("items"))


@app.route("/items/<int:i>/toggle", methods=["POST"])
def items_toggle(i):
    m = dg.load_menu(); cur = m.loc[m.id == i, "status"].iloc[0]
    m.loc[m.id == i, "status"] = "Inactive" if cur == "Active" else "Active"; dg.save_menu(m)
    return redirect(url_for("items"))


# ------------------------------------------------------------------ Bills
@app.route("/bills")
def bills():
    o, l = dg.load_orders()
    recent = o.sort_values("datetime", ascending=False).head(25)
    ls = l[l.order_id.isin(recent.order_id)].groupby("order_id").apply(lambda g: ", ".join(f"{r.item_name} ×{r.qty}" for r in g.itertuples()), include_groups=False)
    m = dg.load_menu(); m = m[m.status == "Active"]
    return render_template("bills.html", recent=recent, lines=ls.to_dict(), menu=m.to_dict("records"), now=pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M"))


@app.route("/bills/add", methods=["POST"])
def bills_add():
    ids, qtys = request.form.getlist("item_id"), request.form.getlist("qty")
    try:
        lines = [(int(i), int(q)) for i, q in zip(ids, qtys) if i and q and int(q) > 0]
        row = dg.add_bill(lines, request.form.get("order_type", "Dine in"), request.form.get("payment", "Cash"), float(request.form.get("discount_pct") or 0),
                          when=request.form.get("when") or None, party_size=int(request.form.get("party_size") or 1), tip=float(request.form.get("tip") or 0))
        flash(f"Bill #{row['order_id']} saved — total ₹{row['total']:,.2f}", "ok")
    except Exception as e:
        flash(f"Could not save bill: {e}", "err")
    return redirect(url_for("bills"))


# ------------------------------------------------------------------ Sales (image 4)
@app.route("/sales")
def sales():
    d, lines = an.prepare(); this, last, end = an.split_weeks(d)
    pay = d.groupby("payment").agg(tx=("order_id", "count"), amount=("total", "sum"), tips=("tip", "sum")).sort_values("tx", ascending=False)
    pay["total"] = pay.amount + pay.tips
    li = lines.merge(d[["order_id", "date"]], on="order_id").assign(r=lambda x: x.qty * x.price)
    cat_this = li[li.date > end - pd.Timedelta(days=7)].groupby("category").r.sum().sort_values(ascending=False).head(5)
    cat_last = li[(li.date <= end - pd.Timedelta(days=7)) & (li.date > end - pd.Timedelta(days=14))].groupby("category").r.sum().sort_values(ascending=False).head(5)
    lab = lambda a, b: f"{a.strftime('%b %d')} - {b.strftime('%b %d, %Y')}"
    return render_template("sales.html", pay=pay, cat_this=list(cat_this.items()), cat_last=list(cat_last.items()),
                           L1=lab(end - pd.Timedelta(days=6), end), L2=lab(end - pd.Timedelta(days=13), end - pd.Timedelta(days=7)),
                           tot_tx=int(pay.tx.sum()), tot_amt=pay.amount.sum(), tot_tips=pay.tips.sum(), tot=pay.total.sum())


@app.route("/chart/<name>.png")
def chart(name):
    d, lines = an.prepare()
    c = an.sales_charts(d, lines)
    if name not in c: abort(404)
    return send_file(io.BytesIO(c[name]), mimetype="image/png")


# ------------------------------------------------------------------ Analysis (report requirements)
@app.route("/analysis")
def analysis():
    s, e = rng_args(); R = an.analyse(s, e)
    t = lambda df, **k: df.to_html(classes="tbl", border=0, **k)
    fmtd = lambda m: [dict(n=n, b=b, se=se, t=tt, p=p, lo=ci[0], hi=ci[1]) for n, b, se, tt, p, ci in zip(m["names"], m["beta"], m["se"], m["t"], m["p"], m["ci"])]
    return render_template("analysis.html", R=R, s=s or "", e=e or "", describe=t(R["describe"]), by_type=t(R["by_type"]), by_pay=t(R["by_pay"]), by_week=t(R["by_weekend"]),
                           by_cat=t(R["by_cat"].rename(columns={"revenue": "revenue (₹)"})), c1=fmtd(R["m1"]), c2=fmtd(R["m2"]), c4=fmtd(R["m4"]), figs=FIG_TITLES, corr=t(R["corr"]),
                           n=len(R["d"]), top_items=R["by_item"].head(8))


@app.route("/fig/<name>.png")
def fig(name):
    s, e = rng_args(); R = an.analyse(s, e)
    if name not in R["figs"]: abort(404)
    return send_file(io.BytesIO(R["figs"][name]), mimetype="image/png")


# ------------------------------------------------------------------ Export (CSV / PDF)
@app.route("/export")
def export():
    s, e = rng_args(); d, _ = an.prepare(s, e)
    return render_template("export.html", s=s or "", e=e or "", n=len(d), lo=d.date.min().date() if len(d) else "", hi=d.date.max().date() if len(d) else "")


def _csv(df, name):
    b = io.BytesIO(df.to_csv(index=False).encode("utf-8-sig")); return send_file(b, mimetype="text/csv", as_attachment=True, download_name=name)


@app.route("/download/<kind>.csv")
def download_csv(kind):
    s, e = rng_args(); d, lines = an.prepare(s, e)
    if kind == "orders":
        out = d.drop(columns=["date", "hour", "dow", "is_weekend", "bill_size", "good_rating"]).copy()
    elif kind == "items":
        out = lines.assign(revenue=lines.qty * lines.price).groupby(["category", "item_name"], as_index=False).agg(units=("qty", "sum"), revenue=("revenue", "sum")).sort_values("revenue", ascending=False)
    elif kind == "daily":
        out = d.groupby("date", as_index=False).agg(orders=("order_id", "count"), net_sales=("net_sales", "sum"), tax=("tax", "sum"), tips=("tip", "sum"), cost=("cost", "sum"), profit=("profit", "sum"), avg_rating=("rating", "mean")).round({"net_sales":2,"tax":2,"tips":2,"cost":2,"profit":2,"avg_rating":2})
    elif kind == "payments":
        out = d.groupby("payment", as_index=False).agg(transactions=("order_id", "count"), amount=("total", "sum"), non_cash_tips=("tip", "sum")); out["total"] = out.amount + out.non_cash_tips
    elif kind == "categories":
        out = lines.assign(revenue=lines.qty * lines.price).groupby("category", as_index=False).agg(units=("qty", "sum"), revenue=("revenue", "sum")).sort_values("revenue", ascending=False)
    elif kind == "hourly":
        out = d.groupby("hour", as_index=False).agg(orders=("order_id", "count"), net_sales=("net_sales", "sum")).round(2)
    else:
        abort(404)
    return _csv(out, f"{kind}_{pd.Timestamp.now():%Y%m%d}.csv")


@app.route("/download/report.pdf")
def download_pdf():
    s, e = rng_args(); R = an.analyse(s, e); d = R["d"]
    extra = an.sales_charts(R["d"], R["lines"])
    period = f"{d.date.min():%d %b %Y} to {d.date.max():%d %b %Y}"
    return send_file(io.BytesIO(build_pdf(R, extra, period)), mimetype="application/pdf", as_attachment=True, download_name=f"restaurant_report_{pd.Timestamp.now():%Y%m%d}.pdf")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
