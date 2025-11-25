"""
Sequential Analysis Demo
Demonstrates the Sequential Analysis frontend components
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.calculators.sequential_analysis import (
    compute_first_order_markov,
    compute_streak_distribution,
    compute_winrate_given_loss_streak,
    find_max_streaks
)
from frontend.components.transition_matrix import create_transition_matrix_heatmap
from frontend.components.streak_distribution import create_streak_distribution_chart
from frontend.components.conditional_streak_chart import create_conditional_streak_chart
from frontend.components.markov_summary_cards import create_markov_summary_cards


def create_sample_trade_data(n_trades=200, win_rate=0.55, momentum=0.1):
    """
    Create sample trade data with configurable win rate and momentum
    
    Parameters:
    -----------
    n_trades : int
        Number of trades to generate
    win_rate : float
        Base win rate (0-1)
    momentum : float
        Momentum effect: positive = wins follow wins, negative = mean reversion
    """
    np.random.seed(42)
    
    trades = []
    current_outcome = 1 if np.random.random() < win_rate else 0
    
    for i in range(n_trades):
        # Adjust probability based on previous outcome (momentum effect)
        if current_outcome == 1:
            # After a win
            prob_win = win_rate + momentum
        else:
            # After a loss
            prob_win = win_rate - momentum
        
        # Clip to valid range
        prob_win = np.clip(prob_win, 0.1, 0.9)
        
        # Generate next outcome
        current_outcome = 1 if np.random.random() < prob_win else 0
        
        trades.append({
            'trade_id': i + 1,
            'trade_success': current_outcome,
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
        })
    
    return pd.DataFrame(trades)


def main():
    """Run sequential analysis demo"""
    print("=" * 80)
    print("Sequential Analysis Frontend Demo")
    print("=" * 80)
    
    # Create sample data with momentum effect
    print("\n1. Creating sample trade data...")
    df = create_sample_trade_data(n_trades=200, win_rate=0.55, momentum=0.1)
    print(f"   Generated {len(df)} trades")
    print(f"   Overall win rate: {df['trade_success'].mean():.1%}")
    
    # Compute Markov transition matrix
    print("\n2. Computing Markov transition matrix...")
    markov_results = compute_first_order_markov(df, target_column='trade_success', conf_level=0.95)
    
    print("\n   Transition Probabilities:")
    print(f"   P(Win | Win):   {markov_results['probs']['P_win_given_win']:.1%}")
    print(f"   P(Loss | Win):  {markov_results['probs']['P_loss_given_win']:.1%}")
    print(f"   P(Win | Loss):  {markov_results['probs']['P_win_given_loss']:.1%}")
    print(f"   P(Loss | Loss): {markov_results['probs']['P_loss_given_loss']:.1%}")
    print(f"   Total transitions: {markov_results['n_transitions']}")
    
    # Compute streak distribution
    print("\n3. Computing streak distribution...")
    streak_results = compute_streak_distribution(df, target_column='trade_success')
    
    print(f"   Max win streak:  {streak_results['max_win_streak']}")
    print(f"   Max loss streak: {streak_results['max_loss_streak']}")
    print(f"   Avg win streak:  {streak_results['avg_win_streak']:.1f}")
    print(f"   Avg loss streak: {streak_results['avg_loss_streak']:.1f}")
    print(f"   Total win streaks:  {len(streak_results['win_streaks'])}")
    print(f"   Total loss streaks: {len(streak_results['loss_streaks'])}")
    
    # Compute conditional win rate after loss streaks
    print("\n4. Computing conditional win rates...")
    conditional_results = compute_winrate_given_loss_streak(
        df, 
        target_column='trade_success',
        max_streak=10,
        conf_level=0.95,
        min_samples=5
    )
    
    print("\n   P(Win | Loss Streak = k):")
    for _, row in conditional_results.iterrows():
        if row['reliable']:
            print(f"   k={int(row['loss_streak_length'])}: {row['win_rate']:.1%} "
                  f"(n={int(row['n_opportunities'])}, CI=[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}])")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    
    # Transition matrix heatmap
    print("   - Transition matrix heatmap")
    fig_transition = create_transition_matrix_heatmap(markov_results, conf_level=0.95)
    
    # Streak distribution chart
    print("   - Streak distribution chart")
    fig_streaks = create_streak_distribution_chart(streak_results)
    
    # Conditional win rate chart
    print("   - Conditional win rate chart")
    fig_conditional = create_conditional_streak_chart(conditional_results)
    
    # Summary cards (HTML component - just verify it creates without error)
    print("   - Summary cards")
    summary_cards = create_markov_summary_cards(markov_results, streak_results)
    
    print("\n6. Visualization components created successfully!")
    print("   - Transition matrix: ", type(fig_transition).__name__)
    print("   - Streak distribution: ", type(fig_streaks).__name__)
    print("   - Conditional chart: ", type(fig_conditional).__name__)
    print("   - Summary cards: ", type(summary_cards).__name__)
    
    # Save figures as HTML
    print("\n7. Saving visualizations to HTML files...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    fig_transition.write_html(f"{output_dir}/transition_matrix.html")
    print(f"   Saved: {output_dir}/transition_matrix.html")
    
    fig_streaks.write_html(f"{output_dir}/streak_distribution.html")
    print(f"   Saved: {output_dir}/streak_distribution.html")
    
    fig_conditional.write_html(f"{output_dir}/conditional_win_rate.html")
    print(f"   Saved: {output_dir}/conditional_win_rate.html")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print("\nKey Insights:")
    
    # Analyze momentum vs mean reversion
    p_win_win = markov_results['probs']['P_win_given_win']
    p_win_loss = markov_results['probs']['P_win_given_loss']
    
    if p_win_win > p_win_loss + 0.05:
        print("✓ MOMENTUM EFFECT: Wins tend to follow wins (positive momentum)")
    elif p_win_loss > p_win_win + 0.05:
        print("✓ MEAN REVERSION: Wins tend to follow losses (recovery effect)")
    else:
        print("✓ INDEPENDENCE: Outcomes appear independent (random walk)")
    
    print(f"\n✓ Risk Assessment: Max loss streak of {streak_results['max_loss_streak']} "
          f"suggests potential for {streak_results['max_loss_streak']} consecutive losses")
    
    print("\nOpen the HTML files in your browser to view interactive visualizations!")


if __name__ == "__main__":
    main()
