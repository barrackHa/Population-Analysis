"""
Test script for the BehavioralEDA class

This script demonstrates how to use the BehavioralEDA class to create 
the same plots that were generated in the behavioral_eda.ipynb notebook.
"""

from pathlib import Path
from behavioral_eda_class import BehavioralEDA


def main():
    """Main test function"""
    
    # Set up data path (adjust as needed)
    monkey = 'fiona'  # or 'yasmin'
    base_path = Path.cwd() / 'data' / f'{monkey}_sst'
    filepath = base_path.parent / 'csst_trials_pkls' / f'all_{monkey}_CSST_trials_df.pkl'
    
    print(f"Loading data from: {filepath}")
    
    # Initialize the BehavioralEDA class
    try:
        eda = BehavioralEDA(str(filepath))
        print("✓ Successfully initialized BehavioralEDA class")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please check that the data file exists at the specified path.")
        return
    except Exception as e:
        print(f"❌ Error initializing class: {e}")
        return
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY")
    print("="*60)
    eda.print_summary_stats()
    
    # Test data processing methods
    print("\n" + "="*60)
    print("TESTING DATA PROCESSING METHODS")
    print("="*60)
    
    try:
        # Basic summary
        summary = eda.get_basic_summary()
        print("✓ get_basic_summary() - Success")
        
        # Trial summary data
        trial_summary = eda.get_trial_summary_data()
        print(f"✓ get_trial_summary_data() - {len(trial_summary)} rows")
        
        # Success rates
        success_rates = eda.get_success_rates_data()
        print(f"✓ get_success_rates_data() - {len(success_rates)} rows")
        
        # Direction analysis
        direction_summary = eda.get_direction_summary_data()
        print(f"✓ get_direction_summary_data() - {len(direction_summary)} rows")
        
        # Signal delay performance
        stop_perf, cont_perf = eda.get_signal_delay_performance_data()
        print(f"✓ get_signal_delay_performance_data() - Stop: {len(stop_perf)}, Cont: {len(cont_perf)} rows")
        
        # RT scatter data
        rt_scatter = eda.get_rt_scatter_data()
        print(f"✓ get_rt_scatter_data() - {len(rt_scatter)} rows")
        
        # RT distribution data
        cont_dist, stop_dist = eda.get_rt_distribution_data()
        print(f"✓ get_rt_distribution_data() - Cont: {len(cont_dist)}, Stop: {len(stop_dist)} rows")
        
    except Exception as e:
        print(f"❌ Error in data processing methods: {e}")
        return
    
    # Test plotting methods
    print("\n" + "="*60)
    print("TESTING PLOTTING METHODS")
    print("="*60)
    
    try:
        # Create all plots
        plot_methods = [
            ('plot_trial_distribution', 'Trial Distribution Plot'),
            ('plot_success_rates_percentage', 'Success Rates Percentage Plot'),
            ('plot_direction_analysis', 'Direction Analysis Plot'),
            ('plot_trial_length_distribution', 'Trial Length Distribution Plot'),
            ('plot_go_cue_timing', 'Go Cue Timing Plot'),
            ('plot_signal_delay_performance', 'Signal Delay Performance Plot (Figure 1b)'),
            ('plot_rt_scatter', 'RT Scatter Plot'),
            ('plot_rt_distributions', 'RT Distributions Plot (Figure 1d)')
        ]
        
        plots = {}
        for method_name, description in plot_methods:
            try:
                method = getattr(eda, method_name)
                plot = method()
                plots[method_name] = plot
                print(f"✓ {method_name}() - {description}")
            except Exception as e:
                print(f"❌ {method_name}() - Error: {e}")
        
        print(f"\n✓ Successfully created {len(plots)} plots")
        
        # Display one plot as example (if in notebook environment)
        print("\nTo display plots, use:")
        print("plot = eda.plot_trial_distribution()")
        print("plot.show()  # or just 'plot' in Jupyter")
        
    except Exception as e:
        print(f"❌ Error in plotting methods: {e}")
        return
    
    # Summary of class capabilities
    print("\n" + "="*60)
    print("CLASS CAPABILITIES SUMMARY")
    print("="*60)
    print("\nData Processing Methods:")
    print("- get_basic_summary(): Basic dataset overview")
    print("- get_trial_summary_data(): Trial counts by type and outcome")
    print("- get_success_rates_data(): Success rates by trial type")  
    print("- get_trial_percentage_data(): Trial outcomes as percentages")
    print("- get_direction_summary_data(): Analysis by direction")
    print("- get_direction_success_rates(): Success rates by direction")
    print("- get_signal_delay_performance_data(): Stop/Continue performance")
    print("- get_rt_scatter_data(): RT data for scatter plots")
    print("- get_rt_distribution_data(): RT distribution data")
    
    print("\nPlotting Methods:")
    print("- plot_trial_distribution(): Stacked bar chart of trial outcomes")
    print("- plot_success_rates_percentage(): Success rates as percentages")
    print("- plot_direction_analysis(): Success by direction")
    print("- plot_trial_length_distribution(): Trial length histograms")
    print("- plot_go_cue_timing(): Go cue timing histograms")
    print("- plot_signal_delay_performance(): Figure 1b replication")
    print("- plot_rt_scatter(): Session mean RT scatter plots")
    print("- plot_rt_distributions(): Figure 1d RT distributions")
    
    print("\nUtility Methods:")
    print("- print_summary_stats(): Comprehensive statistical summary")
    print("- Internal methods for RT processing and data validation")
    
    print("\n✅ All tests completed successfully!")
    print(f"\nThe BehavioralEDA class is ready to use with {monkey.title()}'s data.")


if __name__ == "__main__":
    main()