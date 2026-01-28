#!/usr/bin/env python3
"""Tests for environment variable handling in configuration."""
import os
import pytest
from unittest.mock import patch
from src.main import _default_config, _validate_config


class TestDefaultConfigEnvVars:
    """Test environment variable loading in _default_config()."""

    def test_default_values_when_env_not_set(self):
        """Test that sensible defaults are used when env vars are not set."""
        # Clear any existing env vars
        env_vars_to_clear = ['WSE_DATA_ROOT', 'WSE_OUTPUT_DIR', 'WSE_CAMA_ROOT', 'WSE_GEOID_ROOT']
        with patch.dict(os.environ, {}, clear=True):
            # Remove these specific vars from the environment
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            cfg = _default_config()

            # Default values should be used
            assert cfg['data_root'] == './data'
            assert cfg['output_dir'] == './output'
            # Optional paths should be None when not set
            assert cfg['cama_root'] is None
            assert cfg['geoid_root'] is None

    def test_env_vars_override_defaults(self):
        """Test that environment variables override default values."""
        test_env = {
            'WSE_DATA_ROOT': '/custom/data',
            'WSE_OUTPUT_DIR': '/custom/output',
            'WSE_CAMA_ROOT': '/custom/cama',
            'WSE_GEOID_ROOT': '/custom/geoid',
        }

        with patch.dict(os.environ, test_env, clear=False):
            cfg = _default_config()

            assert cfg['data_root'] == '/custom/data'
            assert cfg['output_dir'] == '/custom/output'
            assert cfg['cama_root'] == '/custom/cama'
            assert cfg['geoid_root'] == '/custom/geoid'

    def test_partial_env_vars(self):
        """Test when only some env vars are set."""
        test_env = {
            'WSE_DATA_ROOT': '/my/data/path',
        }

        # Clear all WSE_ env vars first, then set only the ones we want
        with patch.dict(os.environ, test_env, clear=False):
            # Explicitly remove others
            for var in ['WSE_OUTPUT_DIR', 'WSE_CAMA_ROOT', 'WSE_GEOID_ROOT']:
                os.environ.pop(var, None)

            cfg = _default_config()

            assert cfg['data_root'] == '/my/data/path'
            assert cfg['output_dir'] == './output'  # default
            assert cfg['cama_root'] is None
            assert cfg['geoid_root'] is None

    def test_resolutions_preserved(self):
        """Test that other config options are preserved."""
        cfg = _default_config()

        # Resolutions should still be present
        assert 'resolutions' in cfg
        assert isinstance(cfg['resolutions'], list)
        assert len(cfg['resolutions']) > 0

    def test_validation_config_preserved(self):
        """Test that validation config is preserved."""
        cfg = _default_config()

        assert 'validation' in cfg
        assert 'min_observations' in cfg['validation']
        assert 'check_duplicates' in cfg['validation']


class TestConfigValidation:
    """Test configuration validation warnings."""

    def test_validate_config_warns_on_missing_cama_root(self, caplog):
        """Test that validation warns when cama_root is missing."""
        cfg = {
            'data_root': './data',
            'output_dir': './output',
            'cama_root': None,
            'geoid_root': '/some/path',
        }

        warnings = _validate_config(cfg)

        assert any('cama_root' in w.lower() for w in warnings)

    def test_validate_config_warns_on_missing_geoid_root(self, caplog):
        """Test that validation warns when geoid_root is missing."""
        cfg = {
            'data_root': './data',
            'output_dir': './output',
            'cama_root': '/some/path',
            'geoid_root': None,
        }

        warnings = _validate_config(cfg)

        assert any('geoid_root' in w.lower() for w in warnings)

    def test_validate_config_warns_on_multiple_missing(self):
        """Test that validation warns about all missing required paths."""
        cfg = {
            'data_root': './data',
            'output_dir': './output',
            'cama_root': None,
            'geoid_root': None,
        }

        warnings = _validate_config(cfg)

        assert len(warnings) >= 2
        assert any('cama_root' in w.lower() for w in warnings)
        assert any('geoid_root' in w.lower() for w in warnings)

    def test_validate_config_no_warnings_when_all_set(self):
        """Test that no warnings when all paths are configured."""
        cfg = {
            'data_root': '/path/to/data',
            'output_dir': '/path/to/output',
            'cama_root': '/path/to/cama',
            'geoid_root': '/path/to/geoid',
        }

        warnings = _validate_config(cfg)

        # Should have no warnings about missing required paths
        assert not any('cama_root' in w.lower() for w in warnings)
        assert not any('geoid_root' in w.lower() for w in warnings)

    def test_validate_config_empty_string_treated_as_missing(self):
        """Test that empty strings are treated as missing."""
        cfg = {
            'data_root': './data',
            'output_dir': './output',
            'cama_root': '',
            'geoid_root': '',
        }

        warnings = _validate_config(cfg)

        assert any('cama_root' in w.lower() for w in warnings)
        assert any('geoid_root' in w.lower() for w in warnings)


class TestEnvVarEdgeCases:
    """Test edge cases for environment variable handling."""

    def test_empty_env_var_uses_default(self):
        """Test that empty string env var uses default value."""
        test_env = {
            'WSE_DATA_ROOT': '',
        }

        with patch.dict(os.environ, test_env, clear=False):
            cfg = _default_config()

            # Empty string should fall back to default
            assert cfg['data_root'] == './data'

    def test_whitespace_only_env_var(self):
        """Test that whitespace-only env var is handled."""
        test_env = {
            'WSE_DATA_ROOT': '   ',
        }

        with patch.dict(os.environ, test_env, clear=False):
            cfg = _default_config()

            # Whitespace-only should fall back to default
            assert cfg['data_root'] == './data'

    def test_path_with_spaces(self):
        """Test that paths with spaces are handled correctly."""
        test_env = {
            'WSE_DATA_ROOT': '/path/with spaces/data',
            'WSE_OUTPUT_DIR': '/another path/output',
        }

        with patch.dict(os.environ, test_env, clear=False):
            cfg = _default_config()

            assert cfg['data_root'] == '/path/with spaces/data'
            assert cfg['output_dir'] == '/another path/output'

    def test_relative_path_preserved(self):
        """Test that relative paths from env vars are preserved."""
        test_env = {
            'WSE_DATA_ROOT': '../relative/path',
            'WSE_OUTPUT_DIR': './local/output',
        }

        with patch.dict(os.environ, test_env, clear=False):
            cfg = _default_config()

            assert cfg['data_root'] == '../relative/path'
            assert cfg['output_dir'] == './local/output'
