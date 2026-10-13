
TREE = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": ["H", "I"],
    "E": ["J", "K"],
    "F": ["L", "M"],
    "G": ["N", "O"],
}
VALUES = {"H": 4, "I": 9, "J": 7, "K": 2,
          "L": 3, "M": 8, "N": 6, "O": 1}


def is_terminal(state):
    return state in VALUES


def successors(state, reverse=False):
    children = TREE.get(state, [])
    return children[::-1] if reverse else children


def utility(state):
    return VALUES[state]


def minimax(state, maximizing, counter, reverse=False):
    counter[0] += 1
    if is_terminal(state):
        return utility(state)
    value = float("-inf") if maximizing else float("inf")
    for child in successors(state, reverse):
        result = minimax(child, not maximizing, counter, reverse)
        value = max(value, result) if maximizing else min(value, result)
    return value


def alphabeta(state, alpha, beta, maximizing, counter, reverse=False,
              visited=None, trace=None):
    counter[0] += 1
    if visited is not None:
        visited.append(state)
    if is_terminal(state):
        return utility(state)

    # Кіру кезіндегі шектер сақталады. Жолдар қаралу ретімен орналасады.
    row = {"node": state, "type": "MAX" if maximizing else "MIN",
           "alpha_before": alpha, "beta_before": beta,
           "children": [], "skipped": []}
    if trace is not None:
        trace.append(row)
    value = float("-inf") if maximizing else float("inf")
    children = successors(state, reverse)
    for index, child in enumerate(children):
        result = alphabeta(child, alpha, beta, not maximizing, counter,
                          reverse, visited, trace)
        row["children"].append(f"{child}={result}")
        if maximizing:
            value = max(value, result)
            alpha = max(alpha, value)
        else:
            value = min(value, result)
            beta = min(beta, value)
        if alpha >= beta:
            # Соңғы баладан кейін шарт орындалса, нақты қысқарту жоқ.
            row["skipped"] = children[index + 1:]
            break
    row.update(value=value, alpha_after=alpha, beta_after=beta)
    return value


def all_nodes(state, reverse=False):
    result = [state]
    for child in successors(state, reverse):
        result.extend(all_nodes(child, reverse))
    return result


def evaluate(features, weights):
    if len(features) != len(weights):
        raise ValueError("Белгілер мен салмақтар саны бірдей болуы керек")
    return sum(x * w for x, w in zip(features, weights))


def expected_value(payoffs, probabilities):
    if len(payoffs) != len(probabilities):
        raise ValueError("Нәтижелер мен ықтималдықтар саны сәйкес емес")
    if any(p < 0 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-9:
        raise ValueError("Ықтималдықтар теріс емес және қосындысы 1 болуы керек")
    return sum(value * p for value, p in zip(payoffs, probabilities))


def run_experiment(reverse=False):
    c1, c2 = [0], [0]
    visited, trace = [], []
    mm = minimax("A", True, c1, reverse)
    ab = alphabeta("A", float("-inf"), float("inf"), True, c2,
                   reverse, visited, trace)
    skipped = [node for node in all_nodes("A", reverse) if node not in visited]
    return {"reverse": reverse, "minimax": mm, "mm_calls": c1[0],
            "alphabeta": ab, "ab_calls": c2[0], "visited": visited,
            "skipped": skipped, "trace": trace,
            "leaves": sum(is_terminal(node) for node in visited)}


def main():
    for reverse in (False, True):
        result = run_experiment(reverse)
        print("\nБағыт:", "оңнан солға" if reverse else "солдан оңға")
        print(f"Minimax: мәні={result['minimax']}, түйіндер={result['mm_calls']}")
        print(f"Alpha-beta: мәні={result['alphabeta']}, түйіндер={result['ab_calls']}")
        print("Қаралған жапырақтар:", result["leaves"])
        print("Қаралу реті:", " -> ".join(result["visited"]))
        print("Қысқартылған:", ", ".join(result["skipped"]) or "жоқ")
        print("Түйін | Түрі | alpha дейін | beta дейін | Балалар | Мән | Қысқарту")
        for row in result["trace"]:
            print(row["node"], row["type"], row["alpha_before"],
                  row["beta_before"], ", ".join(row["children"]),
                  row["value"], ", ".join(row["skipped"]) or "жоқ", sep=" | ")

    print("\nEval: маржа, сенімділік, қуаттылық")
    states = {"X": (8, 7, 2), "Y": (5, 5, 8)}
    for weights in ((0.5, 0.3, 0.2), (0.3125, 0.1875, 0.5)):
        scores = {name: evaluate(features, weights)
                  for name, features in states.items()}
        print("Салмақтар:", weights, "Бағалар:", scores,
              "Таңдау:", max(scores, key=scores.get))

    print("\nЖеткізу стратегиялары: Minimax және Expectimax")
    payoffs = {"A": (8, 6, 4), "B": (3, 7, 5), "C": (2, 9, 1)}
    probabilities = {"A": (0.2, 0.3, 0.5),
                     "B": (0.2, 0.5, 0.3), "C": (0.7, 0.2, 0.1)}
    worst = {name: min(values) for name, values in payoffs.items()}
    expected = {name: expected_value(values, probabilities[name])
                for name, values in payoffs.items()}
    for name in payoffs:
        print(f"{name}: Minimax={worst[name]}, Expectimax={expected[name]:.1f}")
    print("Minimax таңдауы:", max(worst, key=worst.get))
    print("Expectimax таңдауы:", max(expected, key=expected.get))


if __name__ == "__main__":
    main()
