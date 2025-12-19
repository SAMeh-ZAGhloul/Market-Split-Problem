# Market Split Problem - Example Usage
# Demonstrates how to use the various solvers

import numpy as np
import time

# Import solvers
from pyomo_solver import PyomoMarketSplitSolver
from ortools_solver import ORToolsMarketSplitSolver
from lattice_solver import LatticeBasedSolver
from dwave_solver import DWaveMarketSplitSolver
from qiskit_solver import QiskitMarketSplitSolver

def generate_test_instance(seed=42, m=5, n=30):
    """
    Generate a test instance for Market Split Problem
    
    Args:
        seed: Random seed for reproducibility
        m: Number of products
        n: Number of retailers
        
    Returns:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        true_solution: The underlying solution (for verification)
    """
    rng = np.random.default_rng(seed)
    
    # Generate random matrix A (retailer demands for each product)
    A = rng.integers(1, 10, size=(m, n))
    
    # Generate true solution (which retailers to select)
    true_solution = rng.integers(0, 2, size=n)
    
    # Calculate target allocation b = A @ true_solution
    b = A @ true_solution
    
    return A, b, true_solution

def verify_solution(A, b, solution):
    """
    Verify if a solution is correct
    
    Args:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        solution: Solution dictionary with 'x' key
        
    Returns:
        Boolean indicating if solution is correct
    """
    x = solution['x']
    
    # Check if solution is binary
    if not all(val in [0, 1] for val in x):
        return False, "Solution is not binary"
    
    # Check if constraints are satisfied
    for i in range(A.shape[0]):
        actual = sum(A[i, j] * x[j] for j in range(len(x)))
        if actual != b[i]:
            return False, f"Constraint {i} violated: {actual} != {b[i]}"
    
    return True, "Solution is correct"

def run_example():
    """Run example demonstrating different solvers"""
    print("Market Split Problem - Example Usage")
    print("=" * 50)
    
    # Generate test instance
    print("Generating test instance...")
    A, b, true_solution = generate_test_instance(seed=42, m=5, n=30)
    print(f"Problem size: {A.shape[0]} products, {A.shape[1]} retailers")
    print(f"Target allocation: {b}")
    print()
    
    # Test different solvers
    solvers = [
        ("Pyomo + Gurobi", PyomoMarketSplitSolver()),
        ("OR-Tools CP-SAT", ORToolsMarketSplitSolver()),
        ("Lattice-Based", LatticeBasedSolver()),
        ("D-Wave (SA)", DWaveMarketSplitSolver()),
        ("Qiskit VQE", QiskitMarketSplitSolver(method='vqe')),
        ("Qiskit QAOA", QiskitMarketSplitSolver(method='qaoa'))
    ]
    
    results = []
    
    for solver_name, solver in solvers:
        print(f"Testing {solver_name}...")
        try:
            start_time = time.time()
            solution, solve_time = solver.solve_market_split(A, b, time_limit=60)
            total_time = time.time() - start_time
            
            # Verify solution
            is_valid, message = verify_solution(A, b, solution)
            
            results.append({
                'solver': solver_name,
                'success': True,
                'solve_time': solve_time,
                'total_time': total_time,
                'slack_total': solution.get('slack_total', float('inf')),
                'valid': is_valid,
                'message': message
            })
            
            print(f"  ✓ Success in {solve_time:.3f}s")
            print(f"  Slack total: {solution.get('slack_total', 'N/A')}")
            print(f"  Valid: {is_valid} ({message})")
            
        except Exception as e:
            results.append({
                'solver': solver_name,
                'success': False,
                'error': str(e),
                'solve_time': float('inf'),
                'total_time': float('inf'),
                'slack_total': float('inf'),
                'valid': False,
                'message': "Solver failed"
            })
            
            print(f"  ✗ Failed: {e}")
        
        print()
    
    # Summary
    print("SUMMARY")
    print("-" * 30)
    print(f"{'Solver':<20} {'Success':<8} {'Time (s)':<10} {'Slack':<10} {'Valid':<8}")
    print("-" * 30)
    
   _str:
        success" if result[' = "Yes for result in resultssuccess'] else "No"
        time_str = f"{result['solve_time']:.3f}" if result['solve_time'] != float('inf') else "N/A"
        slack_str = f"{result['slack_total']:.3f}" if result['slack_total'] != float('inf') else "N/A"
        valid_str = "Yes" if result['valid'] else "No"
        
        print(f"{result['solver']:<20} {success_str:<8} {time_str:<10} {slack_str:<10} {valid_str:<8}")
    
    return results

def compare_solvers():
    """Compare performance of different solvers on multiple instances"""
    print("\nComparing solvers on multiple instances...")
    print("=" * 50)
    
    # Generate multiple test instances
    instances = []
    instance_sizes = [(3, 15), (4, 20), (5, 25)]
    
    for i, (m, n) in enumerate(instance_sizes):
        A, b, true_solution = generate_test_instance(seed=42+i, m=m, n=n)
        instances.append((A, b))
        print(f"Instance {i+1}: {m}x{n} (products x retailers)")
    
    print()
    
    # Test a subset of solvers for comparison
    solvers = [
        ("OR-Tools", ORToolsMarketSplitSolver()),
        ("Pyomo", PyomoMarketSplitSolver())
    ]
    
    for solver_name, solver in solvers:
        print(f"Testing {solver_name} on all instances...")
        total_time = 0
        successful = 0
        
        for i, (A, b) in enumerate(instances):
            try:
                start_time = time.time()
                solution, solve_time = solver.solve_market_split(A, b, time_limit=30)
                total_time += solve_time
                
                is_valid, _ = verify_solution(A, b, solution)
                if is_valid:
                    successful += 1
                    
                print(f"  Instance {i+1}: {solve_time:.3f}s - {'✓' if is_valid else '✗'}")
                
            except Exception as e:
                print(f"  Instance {i+1}: Failed - {e}")
        
        print(f"  Summary: {successful}/{len(instances)} successful, avg time: {total_time/len(instances):.3f}s")
        print()

if __name__ == "__main__":
    # Run the main example
    results = run_example()
    
    # Run comparison
    compare_solvers()
    
    print("Example usage completed!")
