from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('GLOP')

var1 = solver.NumVar(0, 1, 'a')
var2 = solver.NumVar(0, solver.infinity(), 'b')

print("Number of variables =", solver.NumVariables())

con = var1 + var2 <= 20
solver.Add(con)

print("Number of constraints =", solver.NumConstraints())

obj = (3 * var1) + var2 - 10

solver.Maximize(obj)
solver.Solve()
print('obj: %s' % solver.Objective().Value())
print('var1: %s' % var1.solution_value())
print('var2: %s' % var2.solution_value())

con2 = var1 <= 0.5
solver.Add(con2)
solver.Add(con2)

print("Number of constraints =", solver.NumConstraints())

solver.Solve()
print('obj: %s' % solver.Objective().Value())
print('var1: %s' % var1.solution_value())
print('var2: %s' % var2.solution_value())
