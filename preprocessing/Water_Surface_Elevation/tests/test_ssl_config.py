#!/usr/bin/env python3
"""Tests for SSL verification configuration in downloader module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings


class TestSSLConfiguration:
    """Test SSL verification configuration."""

    def test_ssl_verification_enabled_by_default(self):
        """SSL verification should be enabled by default for most downloaders."""
        from src.readers.downloader import HydroWebDownloader, HydroSatDownloader

        # HydroWeb should default to True (secure)
        downloader = HydroWebDownloader(output_dir='/tmp/test')
        assert downloader.verify_ssl is True

        # HydroSat is special - defaults to False due to server certificate issues
        hydrosat = HydroSatDownloader(output_dir='/tmp/test')
        assert hydrosat.verify_ssl is False  # Expected: server has cert issues

    def test_ssl_verification_can_be_disabled(self):
        """SSL verification can be explicitly disabled."""
        from src.readers.downloader import HydroSatDownloader

        downloader = HydroSatDownloader(output_dir='/tmp/test', verify_ssl=False)

        assert downloader.verify_ssl is False

    def test_ssl_warning_only_when_disabled(self):
        """SSL warning should only be logged when verification is disabled."""
        from src.readers.downloader import HydroSatDownloader

        # With verify_ssl=True, no warning should be logged
        mock_logger = MagicMock()
        downloader_secure = HydroSatDownloader(
            output_dir='/tmp/test',
            logger=mock_logger,
            verify_ssl=True
        )
        # Check no SSL warning was logged
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if 'SSL' in str(call)
        ]
        assert len(warning_calls) == 0

        # With verify_ssl=False, warning should be logged
        mock_logger_insecure = MagicMock()
        downloader_insecure = HydroSatDownloader(
            output_dir='/tmp/test',
            logger=mock_logger_insecure,
            verify_ssl=False
        )
        # Check SSL warning was logged
        warning_calls = [
            call for call in mock_logger_insecure.warning.call_args_list
            if 'SSL' in str(call)
        ]
        assert len(warning_calls) == 1

    def test_requests_use_configured_ssl_setting(self):
        """requests.get() calls should use the configured verify_ssl setting."""
        from src.readers.downloader import HydroSatDownloader

        # Test with verify_ssl=True
        downloader_secure = HydroSatDownloader(output_dir='/tmp/test', verify_ssl=True)

        with patch('src.readers.downloader.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {'content-length': '100'}
            mock_response.iter_content = MagicMock(return_value=[b'test'])
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Mock the file write operation
            with patch('builtins.open', MagicMock()):
                with patch.object(Path, 'stat') as mock_stat:
                    mock_stat.return_value.st_size = 100
                    downloader_secure._download_file('http://example.com/test.zip', Path('/tmp/test.zip'))

            # Check that verify=True was passed
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs.get('verify') is True

    def test_requests_use_verify_false_when_disabled(self):
        """requests.get() calls should use verify=False when SSL verification is disabled."""
        from src.readers.downloader import HydroSatDownloader

        # Test with verify_ssl=False
        downloader_insecure = HydroSatDownloader(output_dir='/tmp/test', verify_ssl=False)

        with patch('src.readers.downloader.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {'content-length': '100'}
            mock_response.iter_content = MagicMock(return_value=[b'test'])
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Mock the file write operation
            with patch('builtins.open', MagicMock()):
                with patch.object(Path, 'stat') as mock_stat:
                    mock_stat.return_value.st_size = 100
                    downloader_insecure._download_file('http://example.com/test.zip', Path('/tmp/test.zip'))

            # Check that verify=False was passed
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs.get('verify') is False

    def test_no_global_ssl_warning_disable(self):
        """urllib3 warnings should NOT be globally disabled at module import."""
        import importlib
        import sys

        # Remove module from cache to test fresh import
        modules_to_remove = [k for k in sys.modules.keys() if 'downloader' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Import should NOT call urllib3.disable_warnings globally
        with patch('urllib3.disable_warnings') as mock_disable:
            # Force re-import
            import src.readers.downloader
            importlib.reload(src.readers.downloader)

            # Should not be called at module import time
            mock_disable.assert_not_called()

    def test_ssl_warning_suppression_context(self):
        """SSL warnings should only be suppressed during downloads with verify_ssl=False."""
        from src.readers.downloader import HydroSatDownloader
        import urllib3.exceptions

        downloader_insecure = HydroSatDownloader(output_dir='/tmp/test', verify_ssl=False)

        # When verify_ssl=False, InsecureRequestWarning should be suppressed
        # during download but not globally
        with patch('src.readers.downloader.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {'content-length': '100'}
            mock_response.iter_content = MagicMock(return_value=[b'test'])
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # This should work without raising warnings when verify_ssl=False
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with patch('builtins.open', MagicMock()):
                    with patch.object(Path, 'stat') as mock_stat:
                        mock_stat.return_value.st_size = 100
                        downloader_insecure._download_file('http://example.com/test.zip', Path('/tmp/test.zip'))

                # No InsecureRequestWarning should have been raised
                # (because they're suppressed when verify_ssl=False)
                insecure_warnings = [
                    warning for warning in w
                    if issubclass(warning.category, urllib3.exceptions.InsecureRequestWarning)
                ]
                assert len(insecure_warnings) == 0


class TestDataDownloadManagerSSLConfig:
    """Test SSL configuration in DataDownloadManager."""

    def test_ssl_config_passed_to_downloaders(self):
        """DataDownloadManager should pass verify_ssl config to downloaders."""
        from src.readers.downloader import DataDownloadManager

        config = {
            'global_paths': {
                'data_sources': {
                    'hydrosat': {'root': '/tmp/hydrosat'},
                },
            },
            'verify_ssl': False,  # Disable SSL verification
        }

        manager = DataDownloadManager(config=config, logger=None)

        # Check that the downloader was created with verify_ssl=False
        hydrosat_downloader = manager.downloaders.get('hydrosat')
        assert hydrosat_downloader is not None
        assert hydrosat_downloader.verify_ssl is False
