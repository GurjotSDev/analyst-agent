"""
Generate the fake datasets for the eval scenarios, with known problems planted.

Writes data/eval/<scenario>/*.csv. Everything is fake and comes from a fixed
random seed, so every run of this script produces byte-identical files, and
the answer keys in scenario.py stay right.

Plain Python only (csv + random, no pandas), so it runs on the host without
the data libraries the sandbox has.

Run from agent/:
    uv run python evals/make_datasets.py
"""

import csv
import random
from datetime import date, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "eval"

FIRST = ["Ana", "Ben", "Chloe", "Dev", "Ella", "Femi", "Grace", "Hiro", "Ines", "Jack",
         "Kira", "Liam", "Mia", "Noah", "Olu", "Priya", "Quinn", "Rosa", "Sam", "Tara"]
LAST = ["Silva", "Chen", "Okafor", "Novak", "Larsen", "Haddad", "Kim", "Moreau", "Rossi", "Patel"]


def write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" stops Windows adding a blank line between rows.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {path.relative_to(OUT.parent.parent)}: {len(rows):,} rows")


def day(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def money(x: float) -> str:
    return f"${x:,.2f}"   # 1234.5 -> '$1,234.50'


# ---------------------------------------------------------------------------
# Retail: orders that point at customers and products
# ---------------------------------------------------------------------------
def retail() -> None:
    rng = random.Random(101)
    out = OUT / "retail"

    regions = ["North", "South", "East", "West"]
    customers = []
    for cid in range(1, 201):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        customers.append([cid, name, rng.choice(regions),
                          day(rng, date(2022, 1, 1), date(2024, 12, 31)).isoformat()])
    # Planted: 5 names with a trailing space, 4 '-' and 2 'unknown' regions,
    # 3 regions spelled 'north'. Each on different customers.
    picks = rng.sample(range(200), 14)
    for i in picks[:5]:
        customers[i][1] += " "
    for i in picks[5:9]:
        customers[i][2] = "-"
    for i in picks[9:11]:
        customers[i][2] = "unknown"
    for i in picks[11:14]:
        customers[i][2] = "north"
    write(out / "customers.csv", ["customer_id", "name", "region", "signup_date"], customers)

    categories = ["Kitchen", "Garden", "Office", "Toys"]
    products = []
    for n in range(1, 41):
        products.append([f"P{n:03d}", f"{rng.choice(['Basic', 'Pro', 'Max'])} item {n}",
                         rng.choice(categories), money(rng.uniform(4, 1500))])
    # Planted: 4 product names with stray spaces, 1 price 'TBD'.
    picks = rng.sample(range(40), 5)
    for i in picks[:2]:
        products[i][1] = " " + products[i][1]
    for i in picks[2:4]:
        products[i][1] = products[i][1].replace(" item", "  item")   # double space
    products[picks[4]][3] = "TBD"
    write(out / "products.csv", ["product_id", "product_name", "category", "price"], products)

    orders = []
    for n in range(1194):
        d = day(rng, date(2025, 1, 1), date(2025, 12, 31))
        orders.append([10001 + n, rng.randint(1, 200), f"P{rng.randint(1, 40):03d}",
                       rng.randint(1, 5), d.isoformat()])
    # Planted: 50 dates written month first by a second source system. Each
    # chosen row gets a day after the 12th, so the order can be worked out.
    for i in rng.sample(range(1194), 50):
        d = date(2025, rng.randint(1, 12), rng.randint(13, 28))
        orders[i][4] = d.strftime("%m/%d/%Y")
    # Planted: 4 orders for customer 999 and 2 for product P099 (neither exists).
    for n, (cust, prod) in enumerate([(999, None)] * 4 + [(None, "P099")] * 2):
        orders.append([11195 + n, cust or rng.randint(1, 200), prod or f"P{rng.randint(1, 40):03d}",
                       rng.randint(1, 5), day(rng, date(2025, 1, 1), date(2025, 12, 31)).isoformat()])
    # Planted: 6 exact duplicates (a load that was retried), copied from
    # ordinary ISO-dated orders, then everything shuffled.
    normal = [o for o in orders[:1194] if "-" in o[4]]
    orders += [list(o) for o in rng.sample(normal, 6)]
    rng.shuffle(orders)
    write(out / "orders.csv", ["order_id", "customer_id", "product_id", "quantity", "order_date"], orders)


# ---------------------------------------------------------------------------
# Banking: accounts and transactions
# ---------------------------------------------------------------------------
def banking() -> None:
    rng = random.Random(202)
    out = OUT / "banking"

    accounts = []
    ids = rng.sample(range(10000, 99999), 150)
    for n in range(150):
        accounts.append([f"{ids[n]:08d}",   # 8 digits with leading zeros: '00012345'
                         f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                         rng.choice(["Checking", "Savings"]),
                         day(rng, date(2015, 1, 1), date(2024, 12, 31)).isoformat()])
    # Planted: 4 'savings' spellings, 1 account opened in the future.
    picks = rng.sample(range(150), 5)
    for i in picks[:4]:
        accounts[i][2] = "savings"
    accounts[picks[4]][3] = "2031-01-15"
    write(out / "accounts.csv", ["account_id", "holder_name", "account_type", "opened_date"], accounts)

    descriptions = {True: ["ATM withdrawal", "Card payment", "Transfer out", "Bill payment"],
                    False: ["Salary", "Transfer in", "Refund", "Cash deposit"]}
    txns = []
    start = datetime(2025, 1, 1)
    for n in range(3000):
        withdrawal = rng.random() < 0.45            # negative amounts are normal here
        amount = -rng.uniform(5, 800) if withdrawal else rng.uniform(10, 2500)
        when = start + timedelta(seconds=rng.randrange(365 * 24 * 3600))
        txns.append([n + 1, accounts[rng.randrange(150)][0], when.strftime("%Y-%m-%d %H:%M:%S"),
                     f"{amount:.2f}", rng.choice(descriptions[withdrawal])])
    picks = rng.sample(range(3000), 7)
    # Planted: 5 sentinel amounts (an old system's "unknown"), 2 future timestamps.
    for i in picks[:5]:
        txns[i][3] = "999999.99"
    txns[picks[5]][2] = "2027-03-01 09:15:00"
    txns[picks[6]][2] = "2027-08-19 16:40:12"
    # Planted: 3 exact duplicates, copied from ordinary deposits.
    normal = [t for i, t in enumerate(txns) if i not in picks and not t[3].startswith("-")]
    txns += [list(t) for t in rng.sample(normal, 3)]
    rng.shuffle(txns)
    write(out / "transactions.csv", ["txn_id", "account_id", "timestamp", "amount", "description"], txns)


# ---------------------------------------------------------------------------
# Shipments: a European warehouse, dates written day first
# ---------------------------------------------------------------------------
def shipments() -> None:
    rng = random.Random(303)
    origins = ["Rotterdam", "Hamburg", "Lyon", "Milan"]
    rows = []
    for n in range(1, 501):
        # Day AND month both 1-12 on purpose: every date reads both ways, so
        # the order can't be worked out from the data. The owner must say.
        shipped = date(2025, rng.randint(1, 12), rng.randint(1, 12))
        delivered = shipped + timedelta(days=rng.randint(1, 9))
        rows.append([f"S{n:04d}", rng.choice(origins), shipped.strftime("%d/%m/%Y"),
                     delivered.isoformat(), f"{rng.uniform(0.5, 40):.1f} kg",
                     f"{rng.uniform(2, 8):.1f}"])
    picks = rng.sample(range(500), 8)
    # Planted: 3 origins with a trailing space, 4 sensor glitches (-999),
    # 1 impossible delivery date.
    for i in picks[:3]:
        rows[i][1] = "Hamburg "
    for i in picks[3:7]:
        rows[i][5] = "-999.0"
    rows[picks[7]][3] = "2025-02-30"
    write(OUT / "shipments" / "shipments.csv",
          ["shipment_id", "origin", "ship_date", "delivered_date", "weight_kg", "temperature_c"], rows)


# ---------------------------------------------------------------------------
# Clean: a product catalog with nothing wrong (the false-alarm test)
# ---------------------------------------------------------------------------
def clean() -> None:
    rng = random.Random(404)
    rows = []
    for n in range(1, 81):
        rows.append([f"SKU-{n:04d}", f"{rng.choice(['Steel', 'Oak', 'Glass', 'Cotton'])} "
                                     f"{rng.choice(['bowl', 'shelf', 'lamp', 'mat', 'tray'])} {n}",
                     rng.choice(["Kitchen", "Living", "Bedroom", "Outdoor"]),
                     f"{rng.uniform(10, 60):.2f}",
                     day(rng, date(2023, 1, 1), date(2025, 6, 30)).isoformat(),
                     rng.choice(["yes", "no"])])
    write(OUT / "clean" / "catalog.csv",
          ["sku_id", "name", "category", "unit_price", "added_date", "in_stock"], rows)


# ---------------------------------------------------------------------------
# Staffing: contractors and their weekly timesheets, with traps for the
# checks that are easiest to get wrong.
# ---------------------------------------------------------------------------
def staffing() -> None:
    rng = random.Random(505)
    teams = ["Data Engineering", "Cloud Ops", "AI Research", "Security"]
    first = ["Ava", "Liam", "Noah", "Mia", "Zoe", "Omar", "Ivy", "Raj", "Lena", "Kai", "Sara", "Theo",
             "Nina", "Leo", "Ana", "Ben", "Eva", "Max", "Ida", "Tom", "Uma", "Jon", "Pia", "Ray"]
    people = []
    for i, name in enumerate(first, start=1):
        people.append([f"C{i:03d}", f"{name} {rng.choice(LAST)}", teams[i % 4],
                       f"${rng.randint(55, 95) * 10}",
                       date(2025, rng.randint(1, 8), rng.randint(1, 28)).isoformat(),
                       rng.choice(["Canada", "USA", "India", "Germany", "UK"])])
    c = {row[0]: row for row in people}   # columns: id, name, team, day_rate, start_date, country
    # Planted in contractors:
    c["C002"][1] = "  " + c["C002"][1]       # 2 names with stray spaces
    c["C009"][1] += " "
    c["C003"][2] = "cloud ops"                # team variants: 2 x Cloud Ops, 1 x Data Engineering
    c["C007"][2] = "CLOUD OPS"
    c["C010"][2] = "data engineering"
    c["C015"][2] = "Sec"                      # an abbreviation: Security
    c["C005"][3] = "$75,000"                  # a typo, x100; hidden while day_rate is text
    c["C004"][4] = "15.03.2025"               # DD.MM.YYYY: only one reading, so no question
    c["C011"][4] = "2025-02-30"               # an impossible date
    c["C006"][5] = "?"                        # a placeholder (C013's 'n/a' pandas reads as missing)
    c["C013"][5] = "n/a"
    people.append(list(c["C008"]))            # 1 exact duplicate row
    later = list(c["C012"])                   # C012 twice, with different data: keep the first
    later[2], later[3] = "Security", "$880"
    people.append(later)

    projects = ["Watson Migration", "Data Lake", "Fraud Model", "SOC Upgrade"]
    weeks = [date(2026, m, d).isoformat() for m in (3, 4) for d in (2, 9, 16, 23)]
    sheets = []
    for row in people[:24]:                   # 3 weeks per contractor, all in March-April 2026
        for week in rng.sample(weeks, 3):
            sheets.append([5001 + len(sheets), row[0], week, rng.choice([32, 36, 38, 40, 40, 40, 42]),
                           rng.choice(projects)])
    # Planted in timesheets (entry_id = 5001 + position):
    sheets[4][4] = "watson migration"         # project variants
    sheets[11][4] = "DATA LAKE"
    sheets[15][3] = 400                       # a typo for 40; hidden while hours is text
    sheets[20][3] = "N/A"                     # pandas reads as missing
    sheets[50][3] = "-"                       # a placeholder in a number column
    sheets[25][2] = "2030-03-02"              # the one future week. Every other week is in
                                              # 2026: a guessed cut-off like > '2025-12-31'
                                              # would blank them all
    sheets[30][2] = "03/16/2026"              # month first; 16 > 12, so only one reading
    for week in ("2026-03-09", "2026-03-16"):  # 2 rows for a contractor who doesn't exist
        sheets.append([5001 + len(sheets), "C099", week, 40, "Data Lake"])
    sheets += [list(sheets[40]), list(sheets[41])]   # 2 exact duplicate rows
    write(OUT / "staffing" / "contractors.csv",
          ["contractor_id", "name", "team", "day_rate", "start_date", "country"], people)
    write(OUT / "staffing" / "timesheets.csv",
          ["entry_id", "contractor_id", "week_start", "hours", "project"], sheets)


if __name__ == "__main__":
    print(f"Writing fake eval datasets to {OUT}")
    retail()
    banking()
    shipments()
    clean()
    staffing()