"""
Test script to verify the Neural Population Analysis app functionality
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def test_poisson_process():
    """Test the NonHomogeneousPoissonProcess class"""
    print("Testing NonHomogeneousPoissonProcess...")
    
    # Import our module
    import neural_analysis_app as app
    
    # Create process instance
    process = app.NonHomogeneousPoissonProcess()
    
    # Test rate function
    t = np.array([-100, 0, 50, 200, 350])
    rates = process.rate_function(t, 0.005, 0.015)
    expected = [0.005, 0.015, 0.015, 0.005, 0.005]  # baseline, enhanced, enhanced, baseline, baseline
    
    print(f"Time points: {t}")
    print(f"Expected rates: {expected}")
    print(f"Actual rates: {rates}")
    
    assert np.allclose(rates, expected), f"Rate function test failed: expected {expected}, got {rates}"
    print("✓ Rate function test passed")
    
    # Test spike generation
    spike_trains = process.generate_spikes(0.005, 0.015, n_trials=5)
    
    assert len(spike_trains) == 5, f"Expected 5 spike trains, got {len(spike_trains)}"
    print(f"✓ Generated {len(spike_trains)} spike trains")
    
    # Check that spikes are within time bounds
    for i, spikes in enumerate(spike_trains):
        if len(spikes) > 0:
            assert np.all(spikes >= -200) and np.all(spikes <= 500), f"Spikes out of bounds in trial {i}"
    
    print("✓ Spike bounds test passed")
    
    return True

def test_neural_app():
    """Test the NeuralAnalysisApp class"""
    print("\nTesting NeuralAnalysisApp...")
    
    import neural_analysis_app as app
    
    # Create app instance
    neural_app = app.NeuralAnalysisApp()
    
    # Test parameter defaults
    assert neural_app.lambda_1 == 0.005, f"Default lambda_1 incorrect: {neural_app.lambda_1}"
    assert neural_app.lambda_2 == 0.015, f"Default lambda_2 incorrect: {neural_app.lambda_2}"
    assert neural_app.n_trials == 20, f"Default n_trials incorrect: {neural_app.n_trials}"
    print("✓ Default parameters test passed")
    
    # Test plot creation (should not raise errors)
    try:
        fig = neural_app.create_plots()
        print("✓ Plot creation test passed")
    except Exception as e:
        print(f"✗ Plot creation failed: {e}")
        return False
    
    # Test stats generation
    try:
        stats = neural_app.get_stats_text()
        assert "Simulation Statistics" in stats, "Stats text missing expected content"
        print("✓ Statistics generation test passed")
    except Exception as e:
        print(f"✗ Statistics generation failed: {e}")
        return False
    
    return True

def test_app_creation():
    """Test app creation without Panel server"""
    print("\nTesting app creation...")
    
    try:
        import neural_analysis_app as app
        # This tests that all imports and class definitions work
        neural_app = app.NeuralAnalysisApp()
        print("✓ App creation test passed")
        return True
    except Exception as e:
        print(f"✗ App creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running Neural Population Analysis App Tests")
    print("=" * 50)
    
    tests = [
        test_poisson_process,
        test_neural_app, 
        test_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())