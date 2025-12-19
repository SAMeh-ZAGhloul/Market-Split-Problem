import pyomo.environ as pyo
from pyomo.opt import SolverFactory

class PyomoMarketSplitSolver:
    def solve_market_split(self, A, b, time_limit=None):
        import time
        start_time = time.time()
        
        m, n = A.shape
        model = pyo.ConcreteModel()
        model.I = pyo.Set(initialize=range(m))
        model.J = pyo.Set(initialize=range(n))
        model.A = pyo.Param(model.I, model.J, initialize={(i,j): A[i,j] for i in range(m) for j in range(n)})
        model.b = pyo.Param(model.I, initialize={i: b[i] for i in range(m)})
        model.x = pyo.Var(model.J, domain=pyo.Binary)
        model.slack_plus = pyo.Var(model.I, domain=pyo.NonNegativeReals)
        model.slack_minus = pyo.Var(model.I, domain=pyo.NonNegativeReals)
        model.objective = pyo.Objective(rule=lambda model: sum(model.slack_plus[i] + model.slack_minus[i] for i in model.I), sense=pyo.Minimize)
        model.balance_constraint = pyo.Constraint(model.I, rule=lambda model, i: sum(model.A[i,j] * model.x[j] for j in model.J) + model.slack_minus[i] - model.slack_plus[i] == model.b[i])

        solver = SolverFactory('gurobi')
        if time_limit:
            solver.options['TimeLimit'] = time_limit
        results = solver.solve(model, tee=False)

        x_solution = [int(pyo.value(model.x[j])) for j in range(n)]
        slack_total = pyo.value(model.objective)
        return {'x': x_solution, 'slack_total': slack_total}, time.time() - start_time
