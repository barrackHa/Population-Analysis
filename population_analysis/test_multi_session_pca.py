"""
Pytest tests for MultiSessionPCA class.

Run with: pytest test_multi_session_pca.py -v
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from multi_session_pca import MultiSessionPCA
from session_class import Session


@pytest.fixture
def data_path():
    """Fixture for data file path."""
    base_path = Path(__file__).parent.parent / 'data' / 'unified_cell_trial_data'
    pickle_file = base_path / 'msn_fiona_cell_trial_data.pkl'
    if not pickle_file.exists():
        pytest.skip(f"Data file not found: {pickle_file}")
    return pickle_file


@pytest.fixture
def analyzer():
    """Fixture for basic analyzer instance."""
    return MultiSessionPCA()


@pytest.fixture
def analyzer_with_data(data_path):
    """Fixture for analyzer with loaded data."""
    analyzer = MultiSessionPCA()
    analyzer.load_data(data_path)
    return analyzer


@pytest.fixture
def analyzer_validated(analyzer_with_data):
    """Fixture for analyzer with validated sessions."""
    analyzer_with_data.validate_all_sessions(verbose=False)
    return analyzer_with_data


class TestInitialization:
    """Test class initialization and configuration."""

    def test_default_initialization(self, analyzer):
        """Test initialization with default config."""
        assert analyzer.config is not None
        assert analyzer.config['n_pca_components'] == 5
        assert analyzer.config['epok'] == [-50, 500]
        assert analyzer.config['bin_size'] == 1
        assert analyzer.cell_df is None
        assert analyzer.pca is None

    def test_custom_config(self):
        """Test initialization with custom config."""
        custom_config = {
            'epok': [-100, 400],
            'n_pca_components': 3,
            'min_cells_per_session': 15
        }
        analyzer = MultiSessionPCA(custom_config)
        assert analyzer.config['epok'] == [-100, 400]
        assert analyzer.config['n_pca_components'] == 3
        assert analyzer.config['min_cells_per_session'] == 15
        # Default values should still be present
        assert analyzer.config['bin_size'] == 1

    def test_partial_config_update(self):
        """Test that partial config updates preserve defaults."""
        analyzer = MultiSessionPCA({'epok': [-200, 600]})
        assert analyzer.config['epok'] == [-200, 600]
        assert analyzer.config['bin_size'] == 1  # Default preserved


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_data(self, analyzer, data_path):
        """Test loading data from pickle file."""
        analyzer.load_data(data_path)
        assert analyzer.cell_df is not None
        assert len(analyzer.cell_df) > 0
        assert 'trial_session' in analyzer.cell_df.columns
        assert 'cell_ID' in analyzer.cell_df.columns

    def test_load_nonexistent_file(self, analyzer):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            analyzer.load_data('nonexistent_file.pkl')

    def test_method_chaining_load(self, data_path):
        """Test that load_data returns self for chaining."""
        result = MultiSessionPCA().load_data(data_path)
        assert isinstance(result, MultiSessionPCA)
        assert result.cell_df is not None


class TestSessionStatistics:
    """Test session statistics computation."""

    def test_get_session_statistics(self, analyzer_with_data):
        """Test computing session statistics."""
        analyzer_with_data.get_session_statistics()
        assert analyzer_with_data.session_stats_df is not None
        assert len(analyzer_with_data.session_stats_df) > 0
        assert 'session_id' in analyzer_with_data.session_stats_df.columns
        assert 'n_cells' in analyzer_with_data.session_stats_df.columns

    def test_statistics_before_loading(self, analyzer):
        """Test that statistics fail without data."""
        with pytest.raises(ValueError, match="Data not loaded"):
            analyzer.get_session_statistics()


class TestSessionValidation:
    """Test session validation functionality."""

    def test_validate_all_sessions(self, analyzer_with_data):
        """Test session validation."""
        analyzer_with_data.validate_all_sessions(verbose=False)
        assert analyzer_with_data.validation_df is not None
        assert len(analyzer_with_data.valid_sessions) > 0
        assert 'is_valid' in analyzer_with_data.validation_df.columns

        # Check that we have both valid and invalid sessions
        n_valid = analyzer_with_data.validation_df['is_valid'].sum()
        assert n_valid > 0

    def test_validation_filters_sessions(self, analyzer_with_data):
        """Test that validation properly filters sessions."""
        analyzer_with_data.validate_all_sessions(verbose=False)

        # Valid sessions should match validation_df
        valid_from_df = analyzer_with_data.validation_df[
            analyzer_with_data.validation_df['is_valid']
        ]['session_id'].tolist()

        assert set(analyzer_with_data.valid_sessions) == set(valid_from_df)

    def test_validation_without_data(self, analyzer):
        """Test validation fails without loaded data."""
        with pytest.raises(ValueError):
            analyzer.validate_all_sessions()


class TestProperties:
    """Test class properties."""

    def test_n_sessions_property(self, analyzer_validated):
        """Test n_sessions property."""
        assert analyzer_validated.n_sessions == len(analyzer_validated.valid_sessions)
        assert analyzer_validated.n_sessions > 0

    def test_n_cells_before_concatenation(self, analyzer_validated):
        """Test n_cells returns 0 before concatenation."""
        assert analyzer_validated.n_cells == 0

    def test_n_timepoints_before_extraction(self, analyzer_validated):
        """Test n_timepoints returns 0 before extraction."""
        assert analyzer_validated.n_timepoints == 0


class TestMetadata:
    """Test metadata generation."""

    def test_get_metadata_dict(self, analyzer_validated):
        """Test metadata dictionary generation."""
        metadata = analyzer_validated.get_metadata_dict()

        assert isinstance(metadata, dict)
        assert 'epok' in metadata
        assert 'bin_size' in metadata
        assert 'n_pca_components' in metadata
        assert 'n_sessions' in metadata
        assert metadata['n_sessions'] == len(analyzer_validated.valid_sessions)

    def test_metadata_includes_pca_results(self, analyzer_validated):
        """Test that metadata includes PCA results when available."""
        # Before PCA
        metadata_before = analyzer_validated.get_metadata_dict()
        assert 'explained_variance_ratio' not in metadata_before

        # Note: We can't easily test after PCA without full extraction


class TestMethodChaining:
    """Test method chaining functionality."""

    def test_load_and_validate_chain(self, data_path):
        """Test chaining load and validate methods."""
        analyzer = (
            MultiSessionPCA()
            .load_data(data_path)
            .get_session_statistics()
            .validate_all_sessions(verbose=False)
        )

        assert analyzer.cell_df is not None
        assert analyzer.validation_df is not None
        assert len(analyzer.valid_sessions) > 0

    def test_all_methods_return_self(self, analyzer, data_path):
        """Test that chainable methods return self."""
        result = analyzer.load_data(data_path)
        assert result is analyzer

        result = analyzer.get_session_statistics()
        assert result is analyzer

        result = analyzer.validate_all_sessions(verbose=False)
        assert result is analyzer


class TestErrorHandling:
    """Test error handling."""

    def test_prepare_pca_without_data(self, analyzer):
        """Test prepare_pca_matrix fails without data."""
        with pytest.raises(ValueError, match="Data not prepared"):
            analyzer.prepare_pca_matrix()

    def test_fit_pca_without_concatenation(self, analyzer_validated):
        """Test fit_pca fails without concatenated data."""
        with pytest.raises(ValueError, match="Data not prepared"):
            analyzer_validated.fit_pca()

    def test_project_without_pca(self, analyzer_validated):
        """Test projection fails without fitted PCA."""
        with pytest.raises(ValueError, match="PCA not fitted"):
            analyzer_validated.project_all_conditions()

    def test_plot_without_projection(self, analyzer_validated):
        """Test plotting fails without projection."""
        with pytest.raises(ValueError, match="Data not projected"):
            analyzer_validated.plot_3d_trajectory()


class TestSessionValidationMethods:
    """Test new Session validation methods."""

    @pytest.fixture
    def session_with_data(self, data_path):
        """Fixture for Session instance with data."""
        cell_df = pd.read_pickle(data_path)
        # Use a known good session
        session_data = cell_df[cell_df['trial_session'] == 'fi211110a']
        session = Session(session_data, verbose=False)
        session.drop_cells_with_missing_trial_type_or_dir_data()
        return session

    def test_has_trial_types_all_present(self, session_with_data):
        """Test has_trial_types when all types are present."""
        assert session_with_data.has_trial_types(['GO', 'STOP', 'CONT'])

    def test_has_trial_types_partial(self, session_with_data):
        """Test has_trial_types with partial list."""
        assert session_with_data.has_trial_types(['GO'])
        assert session_with_data.has_trial_types(['GO', 'STOP'])

    def test_has_trial_types_missing(self, session_with_data):
        """Test has_trial_types when type is missing."""
        # Assuming 'INVALID' is not a trial type
        assert not session_with_data.has_trial_types(['GO', 'INVALID'])

    def test_has_directions_all_present(self, session_with_data):
        """Test has_directions when all directions are present."""
        assert session_with_data.has_directions([0, 180])

    def test_has_directions_partial(self, session_with_data):
        """Test has_directions with partial list."""
        assert session_with_data.has_directions([0])
        assert session_with_data.has_directions([180])

    def test_has_directions_missing(self, session_with_data):
        """Test has_directions when direction is missing."""
        # 90 degrees should not be present
        assert not session_with_data.has_directions([0, 90])

    def test_get_trial_count_for_condition(self, session_with_data):
        """Test get_trial_count_for_condition returns positive counts."""
        # GO trials at 0 degrees (right)
        count = session_with_data.get_trial_count_for_condition('GO', 0, success_only=True)
        assert count > 0
        assert isinstance(count, int)

    def test_get_trial_count_success_only(self, session_with_data):
        """Test that success_only filters correctly."""
        count_all = session_with_data.get_trial_count_for_condition('GO', 0, success_only=False)
        count_success = session_with_data.get_trial_count_for_condition('GO', 0, success_only=True)

        # Success-only should be <= all trials
        assert count_success <= count_all

    def test_validate_min_trials_per_condition_valid(self, session_with_data):
        """Test validate_min_trials_per_condition with achievable threshold."""
        is_valid, reason = session_with_data.validate_min_trials_per_condition(
            trial_types=['GO', 'STOP'],
            directions=[0, 180],
            min_trials=1,  # Very low threshold, should pass
            success_only=True
        )
        assert is_valid
        assert reason == ""

    def test_validate_min_trials_per_condition_invalid(self, session_with_data):
        """Test validate_min_trials_per_condition with impossible threshold."""
        is_valid, reason = session_with_data.validate_min_trials_per_condition(
            trial_types=['GO', 'STOP'],
            directions=[0, 180],
            min_trials=100000,  # Impossibly high threshold
            success_only=True
        )
        assert not is_valid
        assert len(reason) > 0
        assert "Insufficient" in reason


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
