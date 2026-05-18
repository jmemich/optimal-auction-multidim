from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('GLOP')
solver.SetTimeLimit(10)

var1 = solver.NumVar(0, 1, 'a')
var2 = solver.NumVar(0, solver.infinity(), 'b')

print("Number of variables =", solver.NumVariables())

con = var1 + var2 <= 20
solver.Add(con)

print("Number of constraints =", solver.NumConstraints())

obj = (3 * var1) + var2 - 10

con = var1 + var2 <= 20
solver.Add(con)

solver.Maximize(obj)
solver.Solve()

print('[%s] obj: %s' % (20, solver.Objective().Value()))
print('[%s] var1: %s' % (20, var1.solution_value()))
print('[%s] var2: %s' % (20, var2.solution_value()))

r = list(range(20))[::-1]

for i in r:
    con = var1 + var2 <= i
    solver.Add(con)

    solver.Solve()

    print('[%s] obj: %s' % (i, solver.Objective().Value()))
    print('[%s] var1: %s' % (i, var1.solution_value()))
    print('[%s] var2: %s' % (i, var2.solution_value()))
