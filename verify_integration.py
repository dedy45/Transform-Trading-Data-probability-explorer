"""
Integration Verification Script
Verifies that all new features are properly integrated
"""
import os
import sys

def check_file_exists(filepath):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}")
    return exists

def check_import(module_path):
    """Check if a module can be imported"""
    try:
        # Convert file path to module path
        module_name = module_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        __import__(module_name)
        print(f"✅ Import OK: {module_name}")
        return True
    except Exception as e:
        print(f"❌ Import FAILED: {module_name}")
        print(f"   Error: {str(e)}")
        return False

def main():
    print("="*70)
    print("INTEGRATION VERIFICATION")
    print("="*70)
    print()
    
    all_ok = True
    
    # Check new component files
    print("-"*70)
    print("NEW COMPONENT FILES:")
    print("-"*70)
    
    components = [
        'frontend/components/expectancy_dashboard.py',
        'frontend/components/mae_mfe_optimizer.py',
        'frontend/components/monte_carlo_viz.py',
        'frontend/components/composite_score_viz.py',
    ]
    
    for component in components:
        if not check_file_exists(component):
            all_ok = False
    
    print()
    
    # Check modified layout files
    print("-"*70)
    print("MODIFIED LAYOUT FILES:")
    print("-"*70)
    
    layouts = [
        'frontend/layouts/trade_analysis_dashboard_layout.py',
        'frontend/layouts/whatif_scenarios_layout.py',
        'frontend/layouts/probability_explorer_layout.py',
    ]
    
    for layout in layouts:
        if not check_file_exists(layout):
            all_ok = False
    
    print()
    
    # Check backend calculators
    print("-"*70)
    print("BACKEND CALCULATORS:")
    print("-"*70)
    
    calculators = [
        'backend/calculators/expectancy_calculator.py',
        'backend/calculators/mae_mfe_analyzer.py',
        'backend/calculators/monte_carlo_engine.py',
        'backend/calculators/composite_score.py',
    ]
    
    for calculator in calculators:
        if not check_file_exists(calculator):
            all_ok = False
    
    print()
    
    # Check callbacks
    print("-"*70)
    print("CALLBACK FILES:")
    print("-"*70)
    
    callbacks = [
        'frontend/callbacks/expectancy_callbacks.py',
        'frontend/callbacks/mae_mfe_callbacks.py',
        'frontend/callbacks/monte_carlo_callbacks.py',
        'frontend/callbacks/composite_score_callbacks.py',
    ]
    
    for callback in callbacks:
        if not check_file_exists(callback):
            all_ok = False
    
    print()
    
    # Try importing components
    print("-"*70)
    print("IMPORT VERIFICATION:")
    print("-"*70)
    
    try:
        sys.path.insert(0, os.getcwd())
        
        imports_to_check = [
            'frontend.components.expectancy_dashboard',
            'frontend.components.mae_mfe_optimizer',
            'frontend.components.monte_carlo_viz',
            'frontend.components.composite_score_viz',
        ]
        
        for module in imports_to_check:
            if not check_import(module):
                all_ok = False
    except Exception as e:
        print(f"❌ Import verification failed: {str(e)}")
        all_ok = False
    
    print()
    print("="*70)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED - Integration is complete!")
        print()
        print("You can now run the application:")
        print("    python app.py")
        print()
        return True
    else:
        print("❌ SOME CHECKS FAILED - Please review errors above")
        print()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
