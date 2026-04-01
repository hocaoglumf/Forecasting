import numpy as np
import math

# Problem Data
demand = [10, 25, 0, 15, 30, 45, 10, 0, 20, 35, 50, 10]
setup_cost = 200
holding_cost_per_unit_per_week = 2


def print_detailed_table(name, periods, demand, production, inventory):
    """
    Prints the analysis table in a landscape orientation (Periods as columns).
    """
    print(f"\nDetailed Analysis : {name}")

    # Define labels and the data rows
    rows = [
        ("Period", periods),
        ("Demand", demand),
        ("Production", production),
        ("End Inv", inventory)
    ]

    # Calculate column width based on the longest number or label
    col_width = 6
    label_width = 12

    separator = "-" * (label_width + (len(periods) * (col_width + 3)))
    print(separator)

    for label, values in rows:
        row_str = f"{label:<{label_width}} |"
        for val in values:
            row_str += f" {val:>{col_width}} |"
        print(row_str)

    print(separator)


def lot_for_lot(demand, s_cost, h_cost):
    production = [d for d in demand]
    inventory = [0] * len(demand)
    setups = sum(1 for d in demand if d > 0)
    total_cost = setups * s_cost

    print_detailed_table("Lot-for-Lot (L4L)", range(1, len(demand) + 1), demand, production, inventory)
    return total_cost, "L4L"


def eoq_approach(demand, s_cost, h_cost):
    avg_demand = np.mean(demand)
    eoq = round(math.sqrt((2 * avg_demand * s_cost) / h_cost))

    inventory_levels = []
    production_log = []
    current_inv = 0
    total_h_cost = 0
    total_s_cost = 0

    for d in demand:
        prod = 0
        if current_inv < d:
            prod = eoq
            current_inv += eoq
            total_s_cost += s_cost

        current_inv -= d
        production_log.append(prod)
        inventory_levels.append(current_inv)
        total_h_cost += current_inv * h_cost

    print_detailed_table(f"EOQ (Economic Order Quantity) (Q={eoq})", range(1, len(demand) + 1), demand, production_log, inventory_levels)
    return total_s_cost + total_h_cost, f"EOQ (Q={eoq})"


def poq_approach(demand, s_cost, h_cost):
    avg_demand = np.mean(demand)
    eoq = math.sqrt((2 * avg_demand * s_cost) / h_cost)
    interval = max(1, round(eoq / avg_demand))

    production_log = [0] * len(demand)
    inventory_levels = [0] * len(demand)
    total_cost = 0
    current_inv = 0

    i = 0
    while i < len(demand):
        # Determine lot size for the interval
        batch_demand = sum(demand[i: i + interval])
        if batch_demand > 0:
            production_log[i] = batch_demand
            total_cost += s_cost

            temp_inv = batch_demand
            for offset in range(interval):
                idx = i + offset
                if idx < len(demand):
                    temp_inv -= demand[idx]
                    inventory_levels[idx] = temp_inv
                    total_cost += temp_inv * h_cost
        i += interval

    print_detailed_table(f"POQ (Periodic Order Quantity) (T={interval})", range(1, len(demand) + 1), demand, production_log, inventory_levels)
    return total_cost, f"POQ (T={interval})"


# Run Analysis
results = []
results.append(lot_for_lot(demand, setup_cost, holding_cost_per_unit_per_week))
results.append(eoq_approach(demand, setup_cost, holding_cost_per_unit_per_week))
results.append(poq_approach(demand, setup_cost, holding_cost_per_unit_per_week))

print("\n" + "=" * 80)
print(f"{'Summary Technique':<30} | {'Total Cost':<10}")
print("-" * 45)
for cost, name in results:
    print(f"{name:<30} | ${cost:<10.2f}")