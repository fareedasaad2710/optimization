from sa_algorithm import run_sa_multiple_times
results = run_sa_multiple_times(num_runs=10)
print("\n\nFINAL RESULTS FOR REPORT:")
print("="*50)
scores = [r['best_score'] for r in results]
print(f"Best Score:  {min(scores):.4f}")
print(f"Worst Score: {max(scores):.4f}")
print(f"Mean Score:  {sum(scores)/len(scores):.4f}")
print(f"All Scores:  {', '.join([f'{s:.4f}' for s in scores])}")
