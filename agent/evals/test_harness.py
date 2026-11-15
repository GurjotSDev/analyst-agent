"""
End-to-end check of the evaluation harness, with a fake model.

The fake model writes correct code for each message, then answers by
repeating what the sandbox printed. So if every scenario scores 100%:
  1. the harness plumbing works (messages, probe, scoring), and
  2. every answer key in scenario.py matches results computed fresh, in the
     sandbox, from the real files. A typo in a key would show up here.
Then fake models make specific mistakes, to prove each one is marked down.

Needs the sandbox running (docker compose up -d) and the eval datasets
(uv run python evals/make_datasets.py). Run from agent/:
    uv run python evals/test_harness.py
"""

from agent import config
from agent.executor import ExecutorClient
from run_eval import run_once
from scenario import BY_NAME, SCENARIOS

# What a correct bot runs for each scenario: (first message, second message).
# Saves go to output/evals/test/clean/ (see run_once), never output/clean/.
CORRECT = {
    "expenses": ("""
emp = load('/data/employees.csv')
exp = load('/data/expenses.csv')
check(emp, exp)
""", """
emp = merge_variants(emp, 'department')
exp = merge_variants(exp, 'category')
exp = drop_duplicates(exp)
exp = drop_rows(exp, exp['amount'] < 0, reason='refunds')
exp = drop_rows(exp, exp['employee_id'] == 9999, reason='unknown employee')
emp = set_missing(emp, emp['employee_id'] == 1018, 'salary', reason='data error')
exp = fill_missing(exp, 'category', 'Uncategorized')
save_clean(emp)
save_clean(exp)
"""),
    "retail": ("""
customers = load('/data/eval/retail/customers.csv')
products = load('/data/eval/retail/products.csv')
orders = load('/data/eval/retail/orders.csv')
check(customers, products, orders)
""", """
orders = drop_duplicates(orders)
orders = drop_rows(orders, orders['customer_id'] == 999, reason='unknown customer')
orders = drop_rows(orders, orders['product_id'] == 'P099', reason='unknown product')
orders = to_dates(orders, 'order_date', day_first=False)
products = placeholders_to_missing(products, 'price')
products = to_numbers(products, 'price')
customers = placeholders_to_missing(customers, 'region')
customers = merge_variants(customers, 'region')
customers = strip_spaces(customers, 'name')
products = strip_spaces(products, 'product_name')
save_clean(customers)
save_clean(products)
save_clean(orders)
"""),
    "banking": ("""
accounts = load('/data/eval/banking/accounts.csv')
transactions = load('/data/eval/banking/transactions.csv')
check(accounts, transactions)
""", """
transactions = drop_duplicates(transactions)
transactions = set_missing(transactions, transactions['amount'] > 999999, 'amount',
                           reason="old system's placeholder")
accounts = merge_variants(accounts, 'account_type')
save_clean(accounts)
save_clean(transactions)
"""),
    "shipments": ("""
shipments = load('/data/eval/shipments/shipments.csv')
check(shipments)
""", """
shipments = to_dates(shipments, 'ship_date', day_first=True)
with custom_fix(shipments, "remove ' kg' from weights") as fix:
    fix.df['weight_kg'] = fix.df['weight_kg'].str.replace(' kg', '', regex=False)
shipments = fix.df
shipments = to_numbers(shipments, 'weight_kg')
shipments = set_missing(shipments, shipments['temperature_c'] == -999, 'temperature_c',
                        reason='sensor errors')
shipments = strip_spaces(shipments, 'origin')
save_clean(shipments)
"""),
    "clean": ("""
catalog = load('/data/eval/clean/catalog.csv')
check(catalog)
""", """
print('Nothing to apply: no problems were found.')
"""),
    # The conditions are the ones check() hands out (today's date for the
    # future week), as a model that copies them would write.
    "staffing": ("""
contractors = load('/data/eval/staffing/contractors.csv')
timesheets = load('/data/eval/staffing/timesheets.csv')
check(contractors, timesheets)
""", """
contractors = drop_duplicates(contractors)
contractors = drop_rows(contractors, contractors['contractor_id'].duplicated(), reason='keep the first C012')
contractors = strip_spaces(contractors, 'name')
contractors = merge_variants(contractors, 'team')
with custom_fix(contractors, "'Sec' means 'Security'") as fix:
    fix.df.loc[fix.df['team'] == 'Sec', 'team'] = 'Security'
contractors = fix.df
contractors = to_numbers(contractors, 'day_rate')
contractors = set_missing(contractors, (contractors['day_rate'] > 950).fillna(False), 'day_rate', reason='typo')
contractors = to_dates(contractors, 'start_date', invalid='missing')
contractors = placeholders_to_missing(contractors, 'country')
timesheets = drop_duplicates(timesheets)
timesheets = placeholders_to_missing(timesheets, 'hours')
timesheets = to_numbers(timesheets, 'hours')
timesheets = set_missing(timesheets, (timesheets['hours'] > 42).fillna(False), 'hours', reason='typo')
timesheets = to_dates(timesheets, 'week_start', day_first=False)
timesheets = set_missing(timesheets, timesheets['week_start'] > pd.Timestamp.today().normalize(),
                         'week_start', reason='future week')
timesheets = merge_variants(timesheets, 'project')
timesheets = drop_rows(timesheets, ~timesheets['contractor_id'].isin(contractors['contractor_id']),
                       reason='contractor does not exist')
save_clean(contractors)
save_clean(timesheets)
"""),
}


class HonestLLM:
    """Writes the given code once per message, then reports the output."""

    def __init__(self, programs):
        self.programs = list(programs)

    def chat(self, messages: list[dict]) -> str:
        last = messages[-1]["content"]
        if last.startswith("[Execution result]"):
            # Report exactly what the code printed, as the final answer.
            return "Here is what I found:\n" + last.split("Output:", 1)[-1]
        return f"```python\n{self.programs.pop(0)}\n```"


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


def missed(results) -> list[str]:
    return [k for r in results[:-1] for k, v in r["checks"].items() if not v]


executor = ExecutorClient(config.EXECUTOR_URL, config.EXEC_TIMEOUT)
executor.check_sandboxed()

# 1. Correct work scores 100% on every scenario (so every answer key is right).
for scenario in SCENARIOS:
    results = run_once(HonestLLM(CORRECT[scenario.name]), executor, scenario)
    errors = {k: v for r in results[:-1] for k, v in r["probe_errors"].items()}
    total = sum(r["total"] for r in results[:-1])
    check(f"correct work: {scenario.name}", not missed(results), f"{total - len(missed(results))}/{total}"
          + (f" missed {missed(results)} {errors}" if missed(results) else ""))

# 2. No carry-over: the run above saved correct files in the same folder
#    ("test"). A run that never saves must still fail the save check.
expenses = BY_NAME["expenses"]
explore, clean = CORRECT["expenses"]
results = run_once(HonestLLM([explore, clean.replace("save_clean(emp)\nsave_clean(exp)\n", "")]),
                   executor, expenses)
check("earlier runs' files don't count", missed(results) == ["cleaned files saved"], str(missed(results)))

# 3. Mistakes real models make (or could), each caught by the right check.
def mistake(name: str, scenario: str, old: str, new: str, expected: set[str]) -> None:
    explore, clean = CORRECT[scenario]
    assert old in clean, old
    results = run_once(HonestLLM([explore, clean.replace(old, new)]), executor, BY_NAME[scenario])
    check(name, expected <= set(missed(results)), str(missed(results)))

mistake("refunds set to missing, not removed", "expenses",
        "exp = drop_rows(exp, exp['amount'] < 0, reason='refunds')",
        "exp = set_missing(exp, exp['amount'] < 0, 'amount', reason='refunds')",
        {"refunds removed (not NaN)", "5,816 expense rows"})
mistake("unapproved fix: withdrawals removed", "banking",
        "accounts = merge_variants(accounts, 'account_type')",
        "accounts = merge_variants(accounts, 'account_type')\n"
        "transactions = drop_rows(transactions, transactions['amount'] < 0, reason='negatives')",
        {"withdrawals untouched (not approved)", "exactly 3 changes logged"})
mistake("unapproved fix: future dates changed", "banking",
        "accounts = merge_variants(accounts, 'account_type')",
        "accounts = merge_variants(accounts, 'account_type')\n"
        "transactions = to_dates(transactions, 'timestamp')",
        {"timestamps untouched (not approved)", "exactly 3 changes logged"})
mistake("ship dates read month first", "shipments",
        "to_dates(shipments, 'ship_date', day_first=True)",
        "to_dates(shipments, 'ship_date', day_first=False)",
        {"ship dates read day first"})
mistake("guessed future-date cut-off blanks every week", "staffing",
        "timesheets['week_start'] > pd.Timestamp.today().normalize()",
        "timesheets['week_start'] > '2025-12-31'",
        {"ONLY the future week missing"})
mistake("outlier left in (not approved... but asked for)", "staffing",
        "timesheets = set_missing(timesheets, (timesheets['hours'] > 42).fillna(False), 'hours', reason='typo')\n",
        "",
        {"hours numbers, 3 missing"})
mistake("invented fix on clean data", "clean",
        "print('Nothing to apply: no problems were found.')",
        "catalog = to_dates(catalog, 'added_date')",
        {"nothing changed", "catalog unchanged"})