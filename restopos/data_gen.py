"""Data layer: menu, synthetic order history (NumPy), CSV persistence (Pandas), food images (Pillow)."""
import os, re, random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
IMG = os.path.join(BASE, "static", "images")
MENU_CSV = os.path.join(DATA, "menu.csv")
ORDERS_CSV = os.path.join(DATA, "orders.csv")
LINES_CSV = os.path.join(DATA, "lines.csv")
TAX_RATE = 0.05
CATEGORIES = ["Pizza", "Burgers", "Pasta & Mains", "Hot drinks", "Cold drinks", "Dessert"]
CAT_EMOJI = {"Pizza": "🍕", "Burgers": "🍔", "Pasta & Mains": "🍝", "Hot drinks": "☕", "Cold drinks": "🥤", "Dessert": "🍰"}

# name, category, price (INR), cost (INR), description
MENU = [
    ("Classic Margherita", "Pizza", 229, 90, "Fresh mozzarella, ripe tomatoes and basil on a thin crust."),
    ("Pepperoni Supreme", "Pizza", 289, 120, "Loaded with crispy pepperoni slices and gooey cheese."),
    ("Veggie Pizza", "Pizza", 249, 95, "Capsicum, olives, onion and sweet corn with mozzarella."),
    ("Spicy Buffalo Blaze", "Pizza", 299, 125, "Zesty buffalo sauce, jalapeños and creamy ranch."),
    ("Classic Burger", "Burgers", 189, 70, "Juicy patty, lettuce, tomato and house sauce."),
    ("Cheese Burger", "Burgers", 209, 78, "Double cheddar melt over a grilled patty."),
    ("Crispy Chicken Burger", "Burgers", 219, 85, "Buttermilk fried chicken with slaw and mayo."),
    ("Paneer Burger", "Burgers", 199, 72, "Spiced paneer patty with mint chutney."),
    ("Creamy Pasta", "Pasta & Mains", 209, 75, "Penne in a rich white sauce with herbs."),
    ("Arrabbiata Pasta", "Pasta & Mains", 199, 70, "Spicy tomato sauce with garlic and chilli."),
    ("Veg Fried Rice", "Pasta & Mains", 169, 55, "Wok-tossed rice with seasonal vegetables."),
    ("Butter Chicken Bowl", "Pasta & Mains", 279, 110, "Tender chicken in buttery tomato gravy with rice."),
    ("Cappuccino", "Hot drinks", 129, 35, "Espresso with steamed milk and thick foam."),
    ("Masala Chai", "Hot drinks", 59, 14, "Strong tea brewed with ginger and cardamom."),
    ("Hot Chocolate", "Hot drinks", 149, 45, "Dark chocolate melted into creamy milk."),
    ("Filter Coffee", "Hot drinks", 79, 20, "South Indian decoction coffee with frothy milk."),
    ("Iced Lemon Tea", "Cold drinks", 99, 25, "Black tea, lemon and mint over ice."),
    ("Cold Coffee", "Cold drinks", 139, 40, "Blended coffee with ice cream and milk."),
    ("Mango Shake", "Cold drinks", 129, 38, "Alphonso mango blended with chilled milk."),
    ("Fresh Lime Soda", "Cold drinks", 69, 16, "Sweet, salted or mixed, served fizzy."),
    ("Chocolate Brownie", "Dessert", 129, 42, "Warm fudge brownie with vanilla ice cream."),
    ("Cheesecake Slice", "Dessert", 169, 60, "Baked New York style with berry compote."),
    ("Gulab Jamun", "Dessert", 89, 24, "Two soft milk dumplings in cardamom syrup."),
    ("Vanilla Sundae", "Dessert", 119, 34, "Vanilla scoops with chocolate sauce and nuts."),
]


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ----------------------------------------------------------------- images
def _canvas(top, bottom, size=(1280, 880)):
    w, h = size
    g = np.linspace(0, 1, h)[:, None, None]
    a, b = np.array(top, float), np.array(bottom, float)
    arr = (a * (1 - g) + b * g) * np.ones((h, w, 3))
    return Image.fromarray(arr.astype("uint8"))


def _circle(d, c, r, fill, outline=None, width=0):
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill, outline=outline, width=width)


def _shadow(img, c, r, dy=26):
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([c[0] - r, c[1] - r * 0.55 + dy, c[0] + r, c[1] + r * 0.55 + dy], fill=(0, 0, 0, 120))
    sh = sh.filter(ImageFilter.GaussianBlur(22))
    img.paste(sh, (0, 0), sh)


def _pizza(img, name, rnd):
    d = ImageDraw.Draw(img); c = (640, 440)
    _shadow(img, c, 400, 10)
    _circle(d, c, 410, (139, 94, 60)); _circle(d, c, 395, (160, 110, 70))
    _circle(d, c, 360, (222, 178, 110)); _circle(d, c, 330, (190, 60, 40)); _circle(d, c, 318, (240, 200, 90))
    for _ in range(40):
        a, r = rnd.uniform(0, 6.28), rnd.uniform(0, 290)
        x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
        _circle(d, (x, y), rnd.randint(14, 30), (252, 226, 130))
    n = name.lower()
    if "pepperoni" in n:
        spots = [(rnd.uniform(0, 6.28), rnd.uniform(40, 270)) for _ in range(16)]
        for a, r in spots:
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            _circle(d, (x, y), 40, (170, 40, 35)); _circle(d, (x, y), 28, (195, 62, 50))
    elif "margherita" in n:
        for _ in range(7):
            a, r = rnd.uniform(0, 6.28), rnd.uniform(60, 250)
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            _circle(d, (x, y), 36, (214, 48, 40)); _circle(d, (x, y), 22, (232, 90, 70))
        for _ in range(9):
            a, r = rnd.uniform(0, 6.28), rnd.uniform(40, 270)
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            d.ellipse([x - 30, y - 16, x + 30, y + 16], fill=(54, 130, 52))
    elif "veggie" in n:
        for col, rad, cnt in [((40, 40, 40), 15, 14), ((120, 40, 140), 12, 12), ((60, 150, 60), 22, 10), ((250, 200, 30), 12, 18)]:
            for _ in range(cnt):
                a, r = rnd.uniform(0, 6.28), rnd.uniform(30, 280)
                x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
                _circle(d, (x, y), rad, col)
    else:  # buffalo blaze
        for _ in range(14):
            a, r = rnd.uniform(0, 6.28), rnd.uniform(40, 270)
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            _circle(d, (x, y), 30, (46, 140, 50)); _circle(d, (x, y), 15, (200, 230, 150))
        for _ in range(12):
            a, r = rnd.uniform(0, 6.28), rnd.uniform(40, 280)
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            d.rectangle([x - 22, y - 16, x + 22, y + 16], fill=(205, 95, 25))
    for k in range(4):  # slice lines
        a = np.pi * k / 4
        d.line([c[0] - 330 * np.cos(a), c[1] - 330 * np.sin(a), c[0] + 330 * np.cos(a), c[1] + 330 * np.sin(a)], fill=(120, 80, 50), width=4)


def _burger(img, name, rnd):
    d = ImageDraw.Draw(img); c = (640, 470)
    _shadow(img, c, 380, 120)
    _circle(d, c, 430, (236, 232, 224), (205, 200, 190), 6)
    x0, x1 = 330, 950
    d.rounded_rectangle([x0, 600, x1, 670], 34, fill=(214, 150, 70))            # bottom bun
    d.rounded_rectangle([x0 - 20, 560, x1 + 20, 612], 20, fill=(100, 160, 60))  # lettuce
    for i in range(10):
        d.ellipse([x0 - 20 + i * 66, 590, x0 + 50 + i * 66, 624], fill=(100, 160, 60))
    d.rounded_rectangle([x0, 520, x1, 585], 28, fill=(88, 52, 34))              # patty
    n = name.lower()
    if "cheese" in n or "classic" in n:
        d.polygon([(x0 - 15, 505), (x1 + 15, 505), (x1 - 30, 560), (x0 + 30, 560)], fill=(250, 190, 40))
    if "paneer" in n:
        d.rounded_rectangle([x0, 520, x1, 585], 28, fill=(238, 214, 150))
    if "chicken" in n:
        d.rounded_rectangle([x0, 520, x1, 585], 28, fill=(206, 140, 56))
    d.rounded_rectangle([x0 + 10, 482, x1 - 10, 520], 14, fill=(206, 50, 40))   # tomato
    d.pieslice([x0 - 10, 250, x1 + 10, 520], 180, 360, fill=(222, 150, 62))     # top bun
    d.pieslice([x0 + 20, 262, x1 - 20, 500], 205, 300, fill=(238, 176, 92))
    for _ in range(26):
        x, y = rnd.randint(x0 + 90, x1 - 90), rnd.randint(300, 440)
        d.ellipse([x - 9, y - 5, x + 9, y + 5], fill=(250, 235, 200))


def _pasta(img, name, rnd):
    d = ImageDraw.Draw(img); c = (640, 440)
    _shadow(img, c, 400, 16)
    _circle(d, c, 410, (244, 242, 238), (210, 205, 198), 8); _circle(d, c, 300, (250, 249, 246), (225, 220, 214), 5)
    n = name.lower()
    if "rice" in n or "bowl" in n:
        _circle(d, c, 270, (248, 244, 232))
        for _ in range(500):
            a, r = rnd.uniform(0, 6.28), rnd.uniform(0, 255)
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            d.ellipse([x - 12, y - 4, x + 12, y + 4], fill=(255, 252, 240) if "butter" in n else (240, 220, 140))
        if "butter" in n:
            d.pieslice([c[0] - 230, c[1] - 230, c[0] + 230, c[1] + 230], 200, 340, fill=(214, 100, 30))
            for _ in range(8):
                x, y = c[0] + rnd.randint(-150, 150), c[1] + rnd.randint(-190, -60)
                d.rounded_rectangle([x - 26, y - 20, x + 26, y + 20], 10, fill=(150, 66, 28))
        else:
            for col in [(70, 160, 70), (230, 90, 50), (250, 200, 40)]:
                for _ in range(22):
                    a, r = rnd.uniform(0, 6.28), rnd.uniform(0, 220)
                    d.rectangle([c[0] + r * np.cos(a), c[1] + r * np.sin(a), c[0] + r * np.cos(a) + 18, c[1] + r * np.sin(a) + 18], fill=col)
    else:
        sauce = (232, 120, 40) if "arrab" in n else (252, 240, 210)
        _circle(d, c, 250, sauce)
        for _ in range(26):
            a, r = rnd.uniform(0, 6.28), rnd.uniform(0, 200)
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            d.rounded_rectangle([x - 46, y - 12, x + 46, y + 12], 12, fill=(238, 205, 120), outline=(214, 170, 80))
        for _ in range(10):
            a, r = rnd.uniform(0, 6.28), rnd.uniform(0, 190)
            x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
            d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=(60, 140, 60))


def _cup(img, name, rnd):
    d = ImageDraw.Draw(img); c = (640, 440)
    _shadow(img, c, 380, 30)
    _circle(d, c, 390, (246, 244, 240), (214, 210, 200), 6)
    d.rounded_rectangle([c[0] + 230, c[1] - 50, c[0] + 380, c[1] + 50], 50, outline=(250, 250, 250), width=34)  # handle
    _circle(d, c, 270, (255, 255, 255), (222, 218, 210), 8)
    n = name.lower()
    liquid = {"chai": (200, 140, 80), "chocolate": (92, 52, 36), "filter": (150, 100, 60)}.get(
        next((k for k in ("chai", "chocolate", "filter") if k in n), ""), (140, 90, 52))
    _circle(d, c, 230, liquid)
    if "cappuccino" in n:
        _circle(d, c, 200, (240, 220, 190)); _circle(d, c, 150, (200, 150, 100))
        d.polygon([(c[0], c[1] - 150), (c[0] + 80, c[1] - 20), (c[0], c[1] + 150), (c[0] - 80, c[1] - 20)], fill=(245, 232, 210))
    elif "hot chocolate" in n:
        _circle(d, c, 150, (248, 240, 232))
        for k in range(8):
            a = k * 0.785
            d.line([c[0], c[1], c[0] + 140 * np.cos(a), c[1] + 140 * np.sin(a)], fill=(92, 52, 36), width=10)
    else:
        _circle(d, c, 200, tuple(min(255, v + 25) for v in liquid))
        _circle(d, (c[0] - 40, c[1] - 40), 60, tuple(min(255, v + 55) for v in liquid))


def _glass(img, name, rnd):
    d = ImageDraw.Draw(img)
    _shadow(img, (640, 760), 230, 0)
    n = name.lower()
    liq = (240, 214, 90) if "lemon" in n or "lime" in n else (220, 170, 120) if "coffee" in n else (250, 170, 30)
    if "lime" in n: liq = (170, 220, 120)
    x0, x1, y0, y1 = 470, 810, 170, 770
    d.polygon([(x0, y0), (x1, y0), (x1 - 34, y1), (x0 + 34, y1)], fill=(235, 245, 250), outline=(255, 255, 255))
    d.polygon([(x0 + 14, y0 + 90), (x1 - 14, y0 + 90), (x1 - 42, y1 - 12), (x0 + 42, y1 - 12)], fill=liq)
    if "mango" in n or "coffee" in n:
        d.rounded_rectangle([x0 + 14, y0 + 70, x1 - 14, y0 + 130], 28, fill=(250, 244, 230))
    for _ in range(7):
        x, y = rnd.randint(x0 + 50, x1 - 90), rnd.randint(y0 + 140, y1 - 120)
        d.rounded_rectangle([x, y, x + 76, y + 76], 12, fill=(255, 255, 255, 180), outline=(235, 245, 250), width=4)
    d.line([x1 - 120, y0 + 30, x1 - 40, y0 - 130], fill=(220, 50, 50), width=16)  # straw
    d.line([x0 + 60, y0 + 30, x0 + 52, y1 - 60], fill=(255, 255, 255), width=12)  # highlight
    if "lemon" in n or "lime" in n:
        _circle(d, (x1 + 10, y0 + 20), 80, (240, 220, 60)); _circle(d, (x1 + 10, y0 + 20), 60, (252, 244, 140))


def _dessert(img, name, rnd):
    d = ImageDraw.Draw(img); c = (640, 460)
    _shadow(img, c, 400, 30)
    _circle(d, c, 410, (244, 242, 238), (210, 205, 198), 8)
    n = name.lower()
    if "brownie" in n:
        d.rounded_rectangle([420, 380, 860, 640], 22, fill=(70, 38, 24)); d.rounded_rectangle([420, 380, 860, 440], 22, fill=(98, 56, 36))
        _circle(d, (640, 340), 120, (252, 244, 220)); _circle(d, (610, 310), 50, (255, 252, 240))
        d.line([430, 480, 850, 470], fill=(120, 70, 44), width=5)
    elif "cheesecake" in n:
        d.polygon([(420, 560), (860, 560), (760, 330), (520, 330)], fill=(250, 236, 200))
        d.rectangle([420, 560, 860, 620], fill=(190, 140, 80))
        for _ in range(14):
            x, y = rnd.randint(540, 740), rnd.randint(340, 400)
            _circle(d, (x, y), 22, (150, 30, 70))
    elif "gulab" in n:
        _circle(d, c, 290, (222, 160, 60))
        for p in [(540, 420), (740, 420), (640, 540)]:
            _circle(d, p, 90, (110, 50, 24)); _circle(d, (p[0] - 20, p[1] - 22), 30, (150, 80, 40))
    else:
        d.polygon([(520, 400), (760, 400), (700, 640), (580, 640)], fill=(236, 245, 250), outline=(255, 255, 255))
        for p, col in [((600, 360), (252, 244, 220)), ((680, 340), (252, 248, 235)), ((640, 300), (255, 252, 240))]:
            _circle(d, p, 70, col)
        d.arc([560, 280, 720, 420], 20, 160, fill=(70, 38, 24), width=22)


PAINT = {"Pizza": _pizza, "Burgers": _burger, "Pasta & Mains": _pasta, "Hot drinks": _cup, "Cold drinks": _glass, "Dessert": _dessert}
BG = {"Pizza": ((52, 44, 40), (28, 24, 22)), "Burgers": ((60, 52, 44), (30, 26, 22)), "Pasta & Mains": ((46, 56, 52), (24, 30, 28)),
      "Hot drinks": ((60, 46, 40), (30, 22, 20)), "Cold drinks": ((36, 54, 62), (18, 26, 32)), "Dessert": ((62, 44, 52), (30, 22, 28))}


def make_image(name, category, path):
    rnd = random.Random(slugify(name))
    img = _canvas(*BG[category]).convert("RGBA")
    PAINT[category](img, name, rnd)
    img.convert("RGB").resize((640, 440), Image.LANCZOS).save(path, "JPEG", quality=90)


# ----------------------------------------------------------------- menu
def ensure_menu():
    os.makedirs(DATA, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    if not os.path.exists(MENU_CSV):
        rows = []
        for i, (n, c, p, k, dsc) in enumerate(MENU, 1):
            fn = slugify(n) + ".jpg"
            rows.append(dict(id=i, name=n, category=c, price=p, cost=k, status="Active", image=fn, description=dsc))
        pd.DataFrame(rows).to_csv(MENU_CSV, index=False)
    m = pd.read_csv(MENU_CSV)
    for _, r in m.iterrows():  # make any missing generated image (user uploads live in static/uploads)
        p = os.path.join(IMG, r["image"])
        if not os.path.exists(p) and not os.path.exists(os.path.join(BASE, "static", "uploads", r["image"])):
            if r["category"] in PAINT: make_image(r["name"], r["category"], p)
    return m


def load_menu():
    ensure_menu()
    return pd.read_csv(MENU_CSV)


def save_menu(m):
    m.to_csv(MENU_CSV, index=False)


def image_url(fn):
    up = os.path.join(BASE, "static", "uploads", fn)
    return f"/static/uploads/{fn}" if os.path.exists(up) else f"/static/images/{fn}"


# ----------------------------------------------------------------- orders (NumPy simulation)
def generate_orders(end="2026-10-11", days=100, seed=42):
    rng = np.random.default_rng(seed)
    menu = load_menu()
    active = menu[menu.status == "Active"].reset_index(drop=True)
    pop = rng.dirichlet(np.ones(len(active)) * 3.0)
    day_w = np.array([0.85, 0.9, 0.95, 1.05, 1.25, 1.45, 1.3])  # Mon..Sun
    hours = np.arange(10, 23)
    hw = np.exp(-((hours - 13.2) ** 2) / 3.0) * 1.0 + np.exp(-((hours - 20.0) ** 2) / 3.6) * 1.15 + 0.08
    hw = hw / hw.sum()
    dates = pd.date_range(end=end, periods=days)
    orders, lines, oid = [], [], 1000
    for dt in dates:
        n = rng.poisson(11.5 * day_w[dt.dayofweek])
        for _ in range(n):
            oid += 1
            hour = int(rng.choice(hours, p=hw)); minute = int(rng.integers(0, 60))
            otype = rng.choice(["Dine in", "Take away", "Delivery"], p=[0.5, 0.3, 0.2])
            party = int(rng.choice([1, 2, 3, 4, 5, 6], p=[.12, .34, .2, .22, .08, .04])) if otype == "Dine in" else 1
            cnt = int(np.clip(round(party * 1.1 + rng.normal(0.9, 0.9)), 1, 9))
            picks = rng.choice(len(active), size=cnt, p=pop)
            idx, qty = np.unique(picks, return_counts=True)
            sub = 0.0; cost = 0.0
            for i, q in zip(idx, qty):
                r = active.iloc[i]
                lines.append((oid, int(r.id), r["name"], r.category, int(q), float(r.price)))
                sub += r.price * q; cost += r.cost * q
            disc = round(sub * rng.choice([0.05, 0.1, 0.15]), 2) if rng.random() < 0.1 else 0.0
            tax = round((sub - disc) * TAX_RATE, 2)
            total = round(sub - disc + tax, 2)
            pay = rng.choice(["UPI", "Card", "Cash", "Wallet"], p=[.4, .25, .3, .05])
            tip = round(sub * rng.uniform(0, 0.1)) if (otype == "Dine in" and pay != "Cash" and rng.random() < 0.5) else 0.0
            peak = 1 if hour in (13, 20) else 0
            weekend = 1 if dt.dayofweek >= 5 else 0
            prep = max(4.0, 8 + 2.2 * cnt + 3 * peak + 2 * weekend + (4 if otype == "Delivery" else 0) + rng.normal(0, 2.5))
            rating = float(np.clip(4.7 - 0.08 * max(prep - 18, 0) + (0.15 if disc else 0) + rng.normal(0, 0.45), 1, 5))
            orders.append(dict(order_id=oid, datetime=dt.replace(hour=hour, minute=minute).strftime("%Y-%m-%d %H:%M"),
                               order_type=otype, payment=pay, party_size=party, items_count=int(qty.sum()),
                               subtotal=float(sub), discount=disc, tax=tax, tip=tip, total=total, cost=float(cost),
                               prep_min=round(prep, 1), rating=round(rating, 1)))
    pd.DataFrame(orders).to_csv(ORDERS_CSV, index=False)
    pd.DataFrame(lines, columns=["order_id", "item_id", "item_name", "category", "qty", "price"]).to_csv(LINES_CSV, index=False)


def load_orders():
    if not os.path.exists(ORDERS_CSV):
        generate_orders()
    o = pd.read_csv(ORDERS_CSV, parse_dates=["datetime"])
    l = pd.read_csv(LINES_CSV)
    return o, l


def add_bill(lines, order_type="Dine in", payment="Cash", discount_pct=0.0, when=None, party_size=1, tip=0.0):
    """lines: list of (item_id, qty). Appends one order (+ its lines) to the CSV store and returns the saved row."""
    o, l = load_orders()
    menu = load_menu().set_index("id")
    oid = int(o.order_id.max()) + 1
    new_lines, sub, cost, cnt = [], 0.0, 0.0, 0
    for item_id, qty in lines:
        qty = int(qty)
        if qty <= 0 or int(item_id) not in menu.index: continue
        r = menu.loc[int(item_id)]
        new_lines.append((oid, int(item_id), r["name"], r["category"], qty, float(r["price"])))
        sub += r["price"] * qty; cost += r["cost"] * qty; cnt += qty
    if not new_lines:
        raise ValueError("Bill has no valid items")
    disc = round(sub * float(discount_pct) / 100, 2)
    tax = round((sub - disc) * TAX_RATE, 2)
    when = pd.Timestamp(when) if when else pd.Timestamp.now().floor("min")
    row = dict(order_id=oid, datetime=when.strftime("%Y-%m-%d %H:%M"), order_type=order_type, payment=payment,
               party_size=int(party_size), items_count=cnt, subtotal=float(sub), discount=disc, tax=tax, tip=float(tip),
               total=round(sub - disc + tax, 2), cost=float(cost), prep_min=np.nan, rating=np.nan)  # prep/rating unknown -> real missing data
    pd.concat([o.assign(datetime=o.datetime.dt.strftime("%Y-%m-%d %H:%M")), pd.DataFrame([row])]).to_csv(ORDERS_CSV, index=False)
    pd.concat([l, pd.DataFrame(new_lines, columns=l.columns)]).to_csv(LINES_CSV, index=False)
    return row
