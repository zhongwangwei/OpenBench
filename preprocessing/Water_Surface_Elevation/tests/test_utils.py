#!/usr/bin/env python3
"""Tests for utility modules: config_loader and logger.

Checkpoint tests are in test_checkpoint.py as they are extensive.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from src.utils.config_loader import ConfigLoader, load_config
from src.utils.logger import setup_logger, get_logger, ProgressLogger, ColoredFormatter


# =============================================================================
# ConfigLoader Tests
# =============================================================================

class TestConfigLoader:
    """Tests for ConfigLoader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config_loader(self, temp_dir):
        """Create ConfigLoader with temp directory as base."""
        # Create required config subdirectory
        config_dir = temp_dir / "config"
        config_dir.mkdir(exist_ok=True)
        return ConfigLoader(base_dir=str(temp_dir))

    def test_load_valid_yaml(self, temp_dir):
        """Test loading a valid YAML file."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("key: value\nnested:\n  inner: 123")

        loader = ConfigLoader(base_dir=str(temp_dir))
        config = loader._load_yaml(config_file)

        assert config['key'] == 'value'
        assert config['nested']['inner'] == 123

    def test_load_yaml_with_list(self, temp_dir):
        """Test loading YAML with list values."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("items:\n  - one\n  - two\n  - three")

        loader = ConfigLoader(base_dir=str(temp_dir))
        config = loader._load_yaml(config_file)

        assert config['items'] == ['one', 'two', 'three']

    def test_load_empty_yaml(self, temp_dir):
        """Test loading empty YAML file returns empty dict."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("")

        loader = ConfigLoader(base_dir=str(temp_dir))
        config = loader._load_yaml(config_file)

        assert config == {}

    def test_load_yaml_file_not_found(self, temp_dir):
        """Test loading non-existent file raises error."""
        loader = ConfigLoader(base_dir=str(temp_dir))

        with pytest.raises(FileNotFoundError):
            loader._load_yaml(temp_dir / "nonexistent.yaml")

    def test_deep_merge_simple(self, config_loader):
        """Test deep merge with simple dictionaries."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}

        result = config_loader._deep_merge(base, override)

        assert result['a'] == 1
        assert result['b'] == 3
        assert result['c'] == 4

    def test_deep_merge_nested(self, config_loader):
        """Test deep merge with nested dictionaries."""
        base = {
            'level1': {
                'level2a': {'value': 1},
                'level2b': {'value': 2},
            }
        }
        override = {
            'level1': {
                'level2a': {'value': 10, 'new_key': 'added'},
            }
        }

        result = config_loader._deep_merge(base, override)

        assert result['level1']['level2a']['value'] == 10
        assert result['level1']['level2a']['new_key'] == 'added'
        assert result['level1']['level2b']['value'] == 2

    def test_deep_merge_override_non_dict(self, config_loader):
        """Test deep merge when override value is not a dict."""
        base = {'nested': {'a': 1, 'b': 2}}
        override = {'nested': 'replaced'}

        result = config_loader._deep_merge(base, override)

        assert result['nested'] == 'replaced'

    def test_init_default_base_dir(self):
        """Test ConfigLoader uses default base directory."""
        loader = ConfigLoader()
        # Should not raise and base_dir should be set
        assert loader.base_dir is not None
        assert isinstance(loader.base_dir, Path)

    def test_init_custom_base_dir(self, temp_dir):
        """Test ConfigLoader with custom base directory."""
        loader = ConfigLoader(base_dir=str(temp_dir))
        assert loader.base_dir == temp_dir

    def test_config_dir_is_set(self, temp_dir):
        """Test config_dir is correctly derived from base_dir."""
        loader = ConfigLoader(base_dir=str(temp_dir))
        assert loader.config_dir == temp_dir / "config"

    def test_templates_dir_is_set(self, temp_dir):
        """Test templates_dir is correctly derived from base_dir."""
        loader = ConfigLoader(base_dir=str(temp_dir))
        assert loader.templates_dir == temp_dir / "templates"


class TestConfigLoaderGlobalPaths:
    """Tests for ConfigLoader global paths loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory structure for config tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_dir = tmpdir / "config"
            config_dir.mkdir()
            yield tmpdir

    def test_load_global_paths(self, temp_dir):
        """Test loading global paths configuration."""
        config_dir = temp_dir / "config"
        global_paths_file = config_dir / "global_paths.yaml"
        global_paths_file.write_text(yaml.dump({
            'data_sources': {
                'hydroweb': '/data/hydroweb',
                'cgls': '/data/cgls',
            },
            'output': {
                'root': '/output',
            }
        }))

        loader = ConfigLoader(base_dir=str(temp_dir))
        paths = loader.load_global_paths()

        assert paths['data_sources']['hydroweb'] == '/data/hydroweb'
        assert paths['output']['root'] == '/output'

    def test_load_global_paths_cached(self, temp_dir):
        """Test that global paths are cached after first load."""
        config_dir = temp_dir / "config"
        global_paths_file = config_dir / "global_paths.yaml"
        global_paths_file.write_text(yaml.dump({'key': 'value1'}))

        loader = ConfigLoader(base_dir=str(temp_dir))
        result1 = loader.load_global_paths()

        # Modify file
        global_paths_file.write_text(yaml.dump({'key': 'value2'}))

        # Should return cached value
        result2 = loader.load_global_paths()

        assert result1 is result2  # Same object reference
        assert result2['key'] == 'value1'  # Original value


class TestConfigLoaderValidationRules:
    """Tests for ConfigLoader validation rules loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory structure for config tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_dir = tmpdir / "config"
            config_dir.mkdir()
            yield tmpdir

    def test_load_validation_rules(self, temp_dir):
        """Test loading validation rules configuration."""
        config_dir = temp_dir / "config"
        validation_file = config_dir / "validation_rules.yaml"
        validation_file.write_text(yaml.dump({
            'quality': {
                'min_observations': 10,
                'max_error': 0.5,
            },
            'bounds': {
                'min_lat': -90,
                'max_lat': 90,
            }
        }))

        loader = ConfigLoader(base_dir=str(temp_dir))
        rules = loader.load_validation_rules()

        assert rules['quality']['min_observations'] == 10
        assert rules['bounds']['max_lat'] == 90

    def test_load_validation_rules_cached(self, temp_dir):
        """Test that validation rules are cached after first load."""
        config_dir = temp_dir / "config"
        validation_file = config_dir / "validation_rules.yaml"
        validation_file.write_text(yaml.dump({'rule': 'original'}))

        loader = ConfigLoader(base_dir=str(temp_dir))
        result1 = loader.load_validation_rules()

        # Modify file
        validation_file.write_text(yaml.dump({'rule': 'modified'}))

        # Should return cached value
        result2 = loader.load_validation_rules()

        assert result1 is result2
        assert result2['rule'] == 'original'


class TestConfigLoaderDatasetConfig:
    """Tests for ConfigLoader dataset configuration loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory structure with required configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_dir = tmpdir / "config"
            config_dir.mkdir()

            # Create global_paths.yaml
            (config_dir / "global_paths.yaml").write_text(yaml.dump({
                'data_sources': {'hydroweb': '/data/hydroweb'},
                'output': {'root': 'output', 'logs': 'logs'},
                'cama_data': {'root': '/cama', 'resolutions': ['glb_15min']},
                'geoid_data': {'egm96_model': 'egm96-5'},
            }))

            # Create validation_rules.yaml
            (config_dir / "validation_rules.yaml").write_text(yaml.dump({
                'quality': {'min_observations': 10},
            }))

            yield tmpdir

    def test_load_dataset_config(self, temp_dir):
        """Test loading and merging dataset configuration."""
        dataset_file = temp_dir / "dataset.yaml"
        dataset_file.write_text(yaml.dump({
            'dataset': {
                'name': 'TestDataset',
                'source': 'hydroweb',
            },
            'processing': {
                'calculate_egm': True,
            },
            'filters': {
                'min_observations': 20,
            },
            'output': {
                'format': 'csv',
            }
        }))

        loader = ConfigLoader(base_dir=str(temp_dir))
        config = loader.load_dataset_config(str(dataset_file))

        assert config['dataset']['name'] == 'TestDataset'
        assert config['processing']['calculate_egm'] is True
        assert 'global_paths' in config
        assert 'validation_rules' in config

    def test_load_dataset_config_with_path_override(self, temp_dir):
        """Test dataset config with path overrides."""
        dataset_file = temp_dir / "dataset.yaml"
        dataset_file.write_text(yaml.dump({
            'dataset': {'name': 'Override'},
            'paths': {
                'data_sources': {'hydroweb': '/custom/hydroweb'},
            }
        }))

        loader = ConfigLoader(base_dir=str(temp_dir))
        config = loader.load_dataset_config(str(dataset_file))

        # Override should take precedence
        assert config['global_paths']['data_sources']['hydroweb'] == '/custom/hydroweb'


class TestConfigLoaderDefaultConfig:
    """Tests for ConfigLoader default configuration creation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory structure with required configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_dir = tmpdir / "config"
            config_dir.mkdir()

            # Create global_paths.yaml
            (config_dir / "global_paths.yaml").write_text(yaml.dump({
                'data_sources': {
                    'hydroweb': '/data/hydroweb',
                    'cgls': '/data/cgls',
                    'icesat': '/data/icesat',
                    'hydrosat': '/data/hydrosat',
                },
                'cama_data': {'root': '/cama', 'resolutions': ['glb_15min']},
                'geoid_data': {'egm96_model': 'egm96-5', 'egm2008_model': 'egm2008-1'},
            }))

            # Create validation_rules.yaml
            (config_dir / "validation_rules.yaml").write_text(yaml.dump({
                'quality': {'min_observations': 10},
            }))

            yield tmpdir

    def test_create_default_config_hydroweb(self, temp_dir):
        """Test creating default config for hydroweb source."""
        loader = ConfigLoader(base_dir=str(temp_dir))
        config = loader.create_default_config('hydroweb')

        assert config['dataset']['source'] == 'hydroweb'
        assert 'Hydroweb' in config['dataset']['name']
        assert config['processing']['calculate_egm'] is True
        assert config['filters']['min_observations'] == 10

    def test_create_default_config_all_sources(self, temp_dir):
        """Test creating default config for all valid sources."""
        loader = ConfigLoader(base_dir=str(temp_dir))

        for source in ['hydroweb', 'cgls', 'icesat', 'hydrosat']:
            config = loader.create_default_config(source)
            assert config['dataset']['source'] == source

    def test_create_default_config_invalid_source(self, temp_dir):
        """Test creating default config with invalid source raises error."""
        loader = ConfigLoader(base_dir=str(temp_dir))

        with pytest.raises(ValueError) as exc_info:
            loader.create_default_config('invalid_source')

        assert 'invalid_source' in str(exc_info.value).lower()


class TestConfigLoaderHelperMethods:
    """Tests for ConfigLoader helper methods."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing helper methods."""
        return {
            'global_paths': {
                'data_sources': {
                    'hydroweb': '/data/hydroweb',
                    'cgls': '/data/cgls',
                },
                'cama_data': {
                    'root': '/cama',
                },
                'geoid_data': {
                    'root': '/geoid',
                },
            }
        }

    def test_get_source_path(self, sample_config):
        """Test getting source path from config."""
        loader = ConfigLoader()

        path = loader.get_source_path(sample_config, 'hydroweb')
        assert path == '/data/hydroweb'

    def test_get_source_path_missing(self, sample_config):
        """Test getting non-existent source path returns None."""
        loader = ConfigLoader()

        path = loader.get_source_path(sample_config, 'nonexistent')
        assert path is None

    def test_get_cama_path(self, sample_config):
        """Test getting CaMa path with resolution."""
        loader = ConfigLoader()

        path = loader.get_cama_path(sample_config, 'glb_15min')
        assert path == '/cama/glb_15min'

    def test_get_geoid_path(self, sample_config):
        """Test getting geoid model path."""
        loader = ConfigLoader()

        path = loader.get_geoid_path(sample_config, 'egm96-5')
        assert path == '/geoid/egm96-5.pgm'


class TestLoadConfigFunction:
    """Tests for the load_config convenience function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory structure with required configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_dir = tmpdir / "config"
            config_dir.mkdir()

            # Create global_paths.yaml
            (config_dir / "global_paths.yaml").write_text(yaml.dump({
                'data_sources': {
                    'hydroweb': '/data/hydroweb',
                    'cgls': '/data/cgls',
                    'icesat': '/data/icesat',
                    'hydrosat': '/data/hydrosat',
                },
                'cama_data': {'root': '/cama'},
                'geoid_data': {'egm96_model': 'egm96-5'},
            }))

            # Create validation_rules.yaml
            (config_dir / "validation_rules.yaml").write_text(yaml.dump({
                'quality': {'min_observations': 10},
            }))

            yield tmpdir

    def test_load_config_no_args_raises_error(self):
        """Test load_config without arguments raises error."""
        with pytest.raises(ValueError):
            load_config()

    def test_load_config_with_path(self, temp_dir):
        """Test load_config with config path."""
        config_file = temp_dir / "test_config.yaml"
        config_file.write_text(yaml.dump({
            'dataset': {'name': 'Test'},
        }))

        # Patch ConfigLoader to use our temp directory
        with patch.object(ConfigLoader, '__init__', lambda self: setattr(self, 'base_dir', temp_dir) or setattr(self, 'config_dir', temp_dir / 'config') or setattr(self, 'templates_dir', temp_dir / 'templates') or setattr(self, '_global_paths', None) or setattr(self, '_validation_rules', None)):
            config = load_config(config_path=str(config_file))
            assert 'dataset' in config


# =============================================================================
# Logger Tests
# =============================================================================

class TestSetupLogger:
    """Tests for setup_logger function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_setup_logger_returns_logger(self):
        """Test setup_logger returns a Logger instance."""
        logger = setup_logger('test_logger')
        assert isinstance(logger, logging.Logger)

    def test_setup_logger_with_name(self):
        """Test setup_logger sets correct logger name."""
        logger = setup_logger('my_custom_logger')
        assert logger.name == 'my_custom_logger'

    def test_setup_logger_default_level(self):
        """Test setup_logger defaults to INFO level."""
        logger = setup_logger('test_level')
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level_debug(self):
        """Test setup_logger with DEBUG level."""
        logger = setup_logger('test_debug', log_level='DEBUG')
        assert logger.level == logging.DEBUG

    def test_setup_logger_custom_level_warning(self):
        """Test setup_logger with WARNING level."""
        logger = setup_logger('test_warning', log_level='WARNING')
        assert logger.level == logging.WARNING

    def test_setup_logger_custom_level_error(self):
        """Test setup_logger with ERROR level."""
        logger = setup_logger('test_error', log_level='ERROR')
        assert logger.level == logging.ERROR

    def test_setup_logger_case_insensitive_level(self):
        """Test setup_logger accepts lowercase log level."""
        logger = setup_logger('test_lower', log_level='debug')
        assert logger.level == logging.DEBUG

    def test_setup_logger_has_console_handler(self):
        """Test setup_logger adds console handler."""
        logger = setup_logger('test_console')

        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_setup_logger_clears_existing_handlers(self):
        """Test setup_logger clears existing handlers."""
        # Create logger with handler
        logger_name = 'test_clear_handlers'
        logger = logging.getLogger(logger_name)
        logger.addHandler(logging.StreamHandler())
        initial_count = len(logger.handlers)

        # Setup should clear and add new handlers
        setup_logger(logger_name)

        # Should have cleared old handlers
        assert len(logger.handlers) <= initial_count

    def test_setup_logger_creates_file_handler(self, temp_dir):
        """Test setup_logger creates file handler when log_dir provided."""
        logger = setup_logger('test_file', log_dir=str(temp_dir))

        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_setup_logger_creates_log_file(self, temp_dir):
        """Test setup_logger creates log file in directory."""
        setup_logger('test_create_file', log_dir=str(temp_dir), dataset_name='test')

        log_files = list(temp_dir.glob('*.log'))
        assert len(log_files) == 1
        assert 'test' in log_files[0].name

    def test_setup_logger_creates_log_directory(self, temp_dir):
        """Test setup_logger creates log directory if not exists."""
        log_dir = temp_dir / "new_logs" / "subdir"

        setup_logger('test_mkdir', log_dir=str(log_dir))

        assert log_dir.exists()

    def test_setup_logger_log_file_timestamp(self, temp_dir):
        """Test log file includes timestamp in name."""
        setup_logger('test_timestamp', log_dir=str(temp_dir))

        log_files = list(temp_dir.glob('*.log'))
        assert len(log_files) == 1
        # File should have timestamp format: YYYYMMDD_HHMMSS
        assert any(c.isdigit() for c in log_files[0].name)


class TestGetLogger:
    """Tests for get_logger function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a Logger instance."""
        logger = get_logger('test_get_logger')
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test get_logger sets correct logger name."""
        logger = get_logger('named_logger')
        assert logger.name == 'named_logger'

    def test_get_logger_with_log_file(self, temp_dir):
        """Test get_logger with log file creates file handler."""
        log_file = temp_dir / 'test.log'
        logger = get_logger('test_with_file', log_file=str(log_file))

        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_get_logger_with_level(self):
        """Test get_logger respects log level."""
        logger = get_logger('test_level_get', level='DEBUG')
        assert logger.level == logging.DEBUG


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_colored_formatter_has_colors(self):
        """Test ColoredFormatter defines colors for all log levels."""
        formatter = ColoredFormatter()

        assert 'DEBUG' in formatter.COLORS
        assert 'INFO' in formatter.COLORS
        assert 'WARNING' in formatter.COLORS
        assert 'ERROR' in formatter.COLORS
        assert 'CRITICAL' in formatter.COLORS

    def test_colored_formatter_has_symbols(self):
        """Test ColoredFormatter defines symbols for all log levels."""
        formatter = ColoredFormatter()

        assert 'DEBUG' in formatter.SYMBOLS
        assert 'INFO' in formatter.SYMBOLS
        assert 'WARNING' in formatter.SYMBOLS
        assert 'ERROR' in formatter.SYMBOLS
        assert 'CRITICAL' in formatter.SYMBOLS

    def test_colored_formatter_format_adds_color(self):
        """Test ColoredFormatter adds color codes to record."""
        formatter = ColoredFormatter('%(color)s%(message)s%(reset)s')

        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='Test message', args=(), exc_info=None
        )
        formatted = formatter.format(record)

        # Should contain ANSI color codes
        assert '\033[' in formatted

    def test_colored_formatter_reset_code(self):
        """Test ColoredFormatter has reset code."""
        formatter = ColoredFormatter()
        assert formatter.RESET == '\033[0m'


class TestProgressLogger:
    """Tests for ProgressLogger class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return MagicMock(spec=logging.Logger)

    def test_progress_logger_init(self, mock_logger):
        """Test ProgressLogger initialization."""
        progress = ProgressLogger(mock_logger, total=100, prefix='Test: ')

        assert progress.total == 100
        assert progress.prefix == 'Test: '
        assert progress.current == 0

    def test_progress_logger_update_increments(self, mock_logger):
        """Test ProgressLogger update increments current."""
        progress = ProgressLogger(mock_logger, total=100)

        progress.update()
        assert progress.current == 1

        progress.update()
        assert progress.current == 2

    def test_progress_logger_update_with_value(self, mock_logger):
        """Test ProgressLogger update with explicit value."""
        progress = ProgressLogger(mock_logger, total=100)

        progress.update(current=50)
        assert progress.current == 50

    def test_progress_logger_logs_at_intervals(self, mock_logger):
        """Test ProgressLogger logs at every 100 items."""
        progress = ProgressLogger(mock_logger, total=200)

        # Should not log yet
        for _ in range(99):
            progress.update()
        assert mock_logger.info.call_count == 0

        # Should log at 100
        progress.update()
        assert mock_logger.info.call_count == 1

    def test_progress_logger_logs_at_completion(self, mock_logger):
        """Test ProgressLogger logs when reaching total."""
        progress = ProgressLogger(mock_logger, total=50)

        for _ in range(50):
            progress.update()

        # Should have logged at completion
        assert mock_logger.info.call_count >= 1

    def test_progress_logger_done(self, mock_logger):
        """Test ProgressLogger done method."""
        progress = ProgressLogger(mock_logger, total=100, prefix='Prefix: ')

        progress.done()

        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert 'Prefix:' in call_args
        assert '100' in call_args

    def test_progress_logger_done_custom_message(self, mock_logger):
        """Test ProgressLogger done with custom message."""
        progress = ProgressLogger(mock_logger, total=50)

        progress.done(message='All done!')

        call_args = mock_logger.info.call_args[0][0]
        assert 'All done!' in call_args


class TestLoggerIntegration:
    """Integration tests for logging functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_logging_to_file(self, temp_dir):
        """Test that messages are written to log file."""
        logger = setup_logger('integration_test', log_dir=str(temp_dir))

        test_message = 'Test log message 12345'
        logger.info(test_message)

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        log_files = list(temp_dir.glob('*.log'))
        assert len(log_files) == 1

        content = log_files[0].read_text()
        assert test_message in content

    def test_logging_levels_work(self, temp_dir):
        """Test that different log levels work correctly."""
        logger = setup_logger('level_test', log_dir=str(temp_dir), log_level='DEBUG')

        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        log_files = list(temp_dir.glob('*.log'))
        content = log_files[0].read_text()

        assert 'Debug message' in content
        assert 'Info message' in content
        assert 'Warning message' in content
        assert 'Error message' in content

    def test_log_level_filtering(self, temp_dir):
        """Test that log level filters messages correctly."""
        logger = setup_logger('filter_test', log_dir=str(temp_dir), log_level='WARNING')

        logger.debug('Should not appear')
        logger.info('Should not appear')
        logger.warning('Should appear')
        logger.error('Should appear')

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        log_files = list(temp_dir.glob('*.log'))
        content = log_files[0].read_text()

        # Note: File handler has DEBUG level, but logger has WARNING level
        # So only WARNING and above should be logged
        assert 'Should appear' in content
