from ortools.sat.python import cp_model
import time

class ORToolsMarketSplitSolver:
    def solve_market_split(self, A, b, time_limit=60):
        start_time = time.time()
        
        m, n = A.shape
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{j}') for j in range(n)]
        slack_plus = [model.NewIntVar(0, 1000, f'slack_plus_{i}') for i in range(m)]
        slack_minus = [model.NewIntVar(0, 1000, f'slack_minus_{i}') for i in range(m)]
        total_slack = sum(slack_plus[i] + slack_minus[i] for i in range(m))
        model.Minimize(total_slack)
        
        for i in range(m):
            contributions = [A[i, j] * x[j] for j in range(n)]
            model.Add(sum(contributions) + slack_minus[i] - slack_plus[i] == b[i])

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)
        
        x_solution = [solver.Value(x[j]) for j in range(n)]
        slack_total = solver.ObjectiveValue()
        return {'x': x_solution, 'slack_total': slack_total}, time.time() - start_time
