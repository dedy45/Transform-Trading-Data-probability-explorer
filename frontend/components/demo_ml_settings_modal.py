"""
Demo: ML Settings Modal Component

This demo shows how to use the ML Settings Modal component.
Run this file to see example usage and test the component.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from frontend.components.ml_settings_modal import (
    load_config,
    get_settings_content,
    save_settings_to_config
)


def demo_load_config():
    """Demo: Load configuration"""
    print("\n" + "="*60)
    print("Demo 1: Load Configuration")
    print("="*60)
    
    config = load_config()
    
    if config:
        print("\n✓ Configuration loaded successfully!")
        print(f"\nSelected features ({len(config['features']['selected'])}):")
        for i, feature in enumerate(config['features']['selected'], 1):
            print(f"  {i}. {feature}")
        
        print(f"\nClassifier hyperparameters:")
        clf_params = config['model_hyperparameters']['classifier']
        print(f"  - n_estimators: {clf_params['n_estimators']}")
        print(f"  - learning_rate: {clf_params['learning_rate']}")
        print(f"  - max_depth: {clf_params['max_depth']}")
        
        print(f"\nQuality thresholds:")
        thresholds = config['thresholds']
        print(f"  A+: prob_win > {thresholds['quality_A_plus']['prob_win_min']}, "
              f"R_P50 > {thresholds['quality_A_plus']['R_P50_min']}")
        print(f"  A:  prob_win > {thresholds['quality_A']['prob_win_min']}, "
              f"R_P50 > {thresholds['quality_A']['R_P50_min']}")
        print(f"  B:  prob_win > {thresholds['quality_B']['prob_win_min']}, "
              f"R_P50 > {thresholds['quality_B']['R_P50_min']}")
    else:
        print("\n✗ Failed to load configuration")


def demo_get_tab_content():
    """Demo: Get content for each tab"""
    print("\n" + "="*60)
    print("Demo 2: Get Tab Content")
    print("="*60)
    
    tabs = {
        'settings-features': 'Features',
        'settings-hyperparams': 'Model Hyperparameters',
        'settings-thresholds': 'Thresholds',
        'settings-display': 'Display'
    }
    
    for tab_id, tab_name in tabs.items():
        content = get_settings_content(tab_id)
        if content:
            print(f"\n✓ {tab_name} tab content created successfully")
        else:
            print(f"\n✗ Failed to create {tab_name} tab content")


def demo_validate_features():
    """Demo: Validate feature selection"""
    print("\n" + "="*60)
    print("Demo 3: Feature Selection Validation")
    print("="*60)
    
    test_cases = [
        (['f1', 'f2', 'f3'], False, "Too few features (3 < 5)"),
        (['f1', 'f2', 'f3', 'f4', 'f5'], True, "Minimum features (5)"),
        (['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'], True, "Valid features (8)"),
        (['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 
          'f11', 'f12', 'f13', 'f14', 'f15'], True, "Maximum features (15)"),
        (['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 
          'f11', 'f12', 'f13', 'f14', 'f15', 'f16'], False, "Too many features (16 > 15)"),
    ]
    
    for features, expected_valid, description in test_cases:
        is_valid = 5 <= len(features) <= 15
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"\n{status} {description}")
        print(f"   Features: {len(features)}, Valid: {is_valid}")


def demo_validate_thresholds():
    """Demo: Validate threshold ordering"""
    print("\n" + "="*60)
    print("Demo 4: Threshold Ordering Validation")
    print("="*60)
    
    test_cases = [
        # (A+ prob, A prob, B prob, A+ R, A R, B R, expected_valid, description)
        (0.65, 0.55, 0.45, 1.5, 1.0, 0.5, True, "Valid ordering (A+ >= A >= B)"),
        (0.55, 0.65, 0.45, 1.5, 1.0, 0.5, False, "Invalid: A+ < A (probability)"),
        (0.65, 0.55, 0.60, 1.5, 1.0, 0.5, False, "Invalid: A < B (probability)"),
        (0.65, 0.55, 0.45, 1.0, 1.5, 0.5, False, "Invalid: A+ < A (R_P50)"),
        (0.65, 0.55, 0.45, 1.5, 1.0, 1.2, False, "Invalid: A < B (R_P50)"),
    ]
    
    for aplus_p, a_p, b_p, aplus_r, a_r, b_r, expected_valid, description in test_cases:
        prob_valid = aplus_p >= a_p >= b_p
        r_valid = aplus_r >= a_r >= b_r
        is_valid = prob_valid and r_valid
        
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"\n{status} {description}")
        print(f"   Prob: A+={aplus_p}, A={a_p}, B={b_p} -> Valid: {prob_valid}")
        print(f"   R_P50: A+={aplus_r}, A={a_r}, B={b_r} -> Valid: {r_valid}")


def demo_settings_structure():
    """Demo: Show settings structure for saving"""
    print("\n" + "="*60)
    print("Demo 5: Settings Structure")
    print("="*60)
    
    settings_dict = {
        'features': {
            'selected': [
                'trend_strength_tf',
                'swing_position',
                'volatility_regime',
                'support_distance',
                'momentum_score'
            ],
            'scaling': 'standard',
            'handle_missing': 'median'
        },
        'model_hyperparameters': {
            'classifier': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'quantile': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_samples': 20,
                'random_state': 42
            }
        },
        'thresholds': {
            'quality_A_plus': {
                'prob_win_min': 0.65,
                'R_P50_min': 1.5
            },
            'quality_A': {
                'prob_win_min': 0.55,
                'R_P50_min': 1.0
            },
            'quality_B': {
                'prob_win_min': 0.45,
                'R_P50_min': 0.5
            },
            'trade_quality_min': 'A'
        },
        'display': {
            'n_bins_reliability': 10,
            'n_top_features': 10,
            'chart_height': 400,
            'chart_template': 'plotly_white',
            'color_scheme': {
                'A_plus': '#006400',
                'A': '#32CD32',
                'B': '#FFD700',
                'C': '#DC143C'
            }
        }
    }
    
    print("\n✓ Example settings structure:")
    print(f"\nFeatures: {len(settings_dict['features']['selected'])} selected")
    print(f"Classifier n_estimators: {settings_dict['model_hyperparameters']['classifier']['n_estimators']}")
    print(f"A+ threshold: prob_win > {settings_dict['thresholds']['quality_A_plus']['prob_win_min']}")
    print(f"Chart height: {settings_dict['display']['chart_height']}px")
    
    print("\n✓ This structure can be passed to save_settings_to_config()")


def demo_usage_workflow():
    """Demo: Show typical usage workflow"""
    print("\n" + "="*60)
    print("Demo 6: Typical Usage Workflow")
    print("="*60)
    
    print("\n1. User opens ML Prediction Engine")
    print("   → Settings loaded from config/ml_prediction_config.yaml")
    
    print("\n2. User clicks 'Settings' button")
    print("   → Modal opens with current settings")
    
    print("\n3. User navigates to 'Features' tab")
    print("   → Feature checklist displayed with current selection")
    
    print("\n4. User selects/deselects features")
    print("   → Real-time validation: count must be 5-15")
    print("   → Color feedback: red if invalid, green if valid")
    
    print("\n5. User navigates to 'Thresholds' tab")
    print("   → Current thresholds displayed")
    
    print("\n6. User adjusts A+ threshold")
    print("   → Input validation: must maintain A+ >= A >= B ordering")
    
    print("\n7. User clicks 'Save Settings'")
    print("   → Validation runs:")
    print("     - Feature count: 5-15")
    print("     - Threshold ordering: A+ >= A >= B")
    print("   → If valid: save to YAML, show success, close modal")
    print("   → If invalid: show error, keep modal open")
    
    print("\n8. Settings saved successfully")
    print("   → Alert: 'Settings saved! May need to retrain models.'")
    print("   → Modal closes")


def run_all_demos():
    """Run all demos"""
    print("\n" + "="*70)
    print(" "*15 + "ML SETTINGS MODAL DEMO")
    print("="*70)
    
    try:
        demo_load_config()
        demo_get_tab_content()
        demo_validate_features()
        demo_validate_thresholds()
        demo_settings_structure()
        demo_usage_workflow()
        
        print("\n" + "="*70)
        print(" "*20 + "✓ All demos completed!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_demos()
