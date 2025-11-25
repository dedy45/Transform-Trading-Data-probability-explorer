"""
Dependency Checker for Trading Probability Explorer
Checks all required packages and provides installation commands
"""
import sys
import subprocess
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def get_package_version(package_name):
    """Get installed package version"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        pass
    return "Unknown"

# Required packages with their import names
REQUIRED_PACKAGES = [
    ('dash', 'dash'),
    ('dash-bootstrap-components', 'dash_bootstrap_components'),
    ('plotly', 'plotly'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('statsmodels', 'statsmodels'),
]

# Optional but recommended packages
OPTIONAL_PACKAGES = [
    ('openpyxl', 'openpyxl'),  # For Excel export
    ('xlsxwriter', 'xlsxwriter'),  # For Excel export
    ('boruta', 'boruta'),  # For Auto Feature Selection
    ('shap', 'shap'),  # For SHAP analysis
    ('catboost', 'catboost'),  # For CatBoost feature importance
    ('xgboost', 'xgboost'),  # For XGBoost feature importance
]

def main():
    print("="*70)
    print("TRADING PROBABILITY EXPLORER - DEPENDENCY CHECK")
    print("="*70)
    print()
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  WARNING: Python 3.8+ is recommended")
    else:
        print("✅ Python version OK")
    print()
    
    # Check required packages
    print("-"*70)
    print("REQUIRED PACKAGES:")
    print("-"*70)
    
    missing_packages = []
    installed_packages = []
    
    for package_name, import_name in REQUIRED_PACKAGES:
        is_installed = check_package(package_name, import_name)
        
        if is_installed:
            version = get_package_version(package_name)
            print(f"✅ {package_name:30s} {version:15s} INSTALLED")
            installed_packages.append(package_name)
        else:
            print(f"❌ {package_name:30s} {'':15s} MISSING")
            missing_packages.append(package_name)
    
    print()
    
    # Check optional packages
    print("-"*70)
    print("OPTIONAL PACKAGES:")
    print("-"*70)
    
    for package_name, import_name in OPTIONAL_PACKAGES:
        is_installed = check_package(package_name, import_name)
        
        if is_installed:
            version = get_package_version(package_name)
            print(f"✅ {package_name:30s} {version:15s} INSTALLED")
        else:
            print(f"⚠️  {package_name:30s} {'':15s} NOT INSTALLED (optional)")
    
    print()
    print("="*70)
    
    # Summary
    if missing_packages:
        print(f"❌ MISSING {len(missing_packages)} REQUIRED PACKAGE(S)")
        print()
        print("To install missing packages, run:")
        print()
        print(f"    pip install {' '.join(missing_packages)}")
        print()
        print("Or install all requirements:")
        print()
        print("    pip install -r requirements.txt")
        print()
        return False
    else:
        print("✅ ALL REQUIRED PACKAGES INSTALLED")
        print()
        print("Application is ready to run!")
        print()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
