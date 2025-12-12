#!/usr/bin/env python3
"""
Test script to run Dragonfly algorithm on Case Study 2
"""

from case_studies import case_study_2_with_dragonfly

if __name__ == "__main__":
    print("Running Dragonfly Algorithm on Case Study 2...")
    results = case_study_2_with_dragonfly()
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)
    print(f"\nBest solution score: {results['best_cost']:.4f}")
    print(f"Runtime: {results['runtime']:.2f} seconds")


