# RestoPOS — Restaurant Menu, Billing, Sales & Analysis

Flask app that applies the DSA report's analysis pipeline (NumPy, Pandas, Matplotlib, SciPy, scikit-learn) to restaurant data.

## Run
```
pip install -r requirements.txt
python app.py          # open http://127.0.0.1:5000
```
First start creates `data/menu.csv`, `data/orders.csv`, `data/lines.csv` (100 days, ~1,300 simulated orders, NumPy seed 42) and the menu images.

## Pages
| Page | URL | What it does |
|---|---|---|
| Dashboard | `/` | Revenue, profit, rating, recent transactions, top sellers, insight card |
| Menu & Billing | `/menu` | POS grid with images, category tabs, search, cart, tax, discount, **Proceed** saves a bill, **Print Receipt** |
| Manage Items | `/items` | Add item (upload image, price, category, status), Edit, Active/Inactive toggle |
| Bills | `/bills` | Add a bill manually, list recent bills |
| Sales | `/sales` | Net sales by time of day / day of week (this week vs last), payment summary, top 5 categories |
| Analysis | `/analysis` | All 12 report figures + tables, missing-data demo, 3 regression models, diagnostics |
| Export | `/export` | CSV (orders, items, daily, payments, categories, hourly) and full PDF report, with date range |

## Files
- `data_gen.py` data store, NumPy order simulation, bill saving, food-image generation
- `analysis.py` pipeline + OLS written in NumPy (coefficients, SE, t, p, F, adj R², VIF, Durbin-Watson) + all figures
- `report_pdf.py` ReportLab PDF · `app.py` routes · `templates/`, `static/`

## Real photos
Upload on **Manage Items → Edit → Replace image** (auto-cropped to one size), or drop `slug-name.jpg` into `static/images/`.
Bills added in the app have blank prep time/rating; analysis treats them as real missing data (median imputation).
