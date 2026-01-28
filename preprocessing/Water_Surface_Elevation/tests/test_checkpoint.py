#!/usr/bin/env python3
"""Tests for secure checkpoint system with HMAC-signed JSON serialization."""

import json
import os
import pickle
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import pytest

from src.utils.checkpoint import Checkpoint, CheckpointManager, CheckpointData


class TestCheckpointSecurity:
    """Test suite for secure checkpoint serialization."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def checkpoint(self, temp_dir):
        """Create a Checkpoint instance."""
        return Checkpoint(str(temp_dir))

    @pytest.fixture
    def sample_results(self):
        """Sample results data for testing."""
        return {
            'stations': ['STN001', 'STN002', 'STN003'],
            'count': 3,
            'metadata': {
                'source': 'hydroweb',
                'timestamp': '2024-01-01T00:00:00',
            },
        }

    # =========================================================================
    # Test 1: JSON Serialization (no pickle)
    # =========================================================================

    def test_save_uses_json_not_pickle(self, checkpoint, sample_results):
        """Verify checkpoint saves data as JSON, not pickle."""
        checkpoint.save('test_step', sample_results)

        checkpoint_file = checkpoint.output_dir / 'checkpoint.json'
        assert checkpoint_file.exists(), "Checkpoint should save as .json file"

        # Should be valid JSON
        with open(checkpoint_file, 'r') as f:
            content = f.read()

        # The file should contain JSON with signature
        parsed = json.loads(content)
        assert 'data' in parsed, "JSON should contain 'data' field"
        assert 'signature' in parsed, "JSON should contain 'signature' field"

    def test_no_pickle_file_created(self, checkpoint, sample_results):
        """Ensure no .pkl file is created."""
        checkpoint.save('test_step', sample_results)

        pkl_file = checkpoint.output_dir / 'checkpoint.pkl'
        assert not pkl_file.exists(), "Should not create .pkl file"

    # =========================================================================
    # Test 2: HMAC Signature Verification
    # =========================================================================

    def test_signature_is_valid_hmac(self, checkpoint, sample_results):
        """Verify signature is a valid HMAC-SHA256 hex digest."""
        checkpoint.save('test_step', sample_results)

        checkpoint_file = checkpoint.output_dir / 'checkpoint.json'
        with open(checkpoint_file, 'r') as f:
            parsed = json.loads(f.read())

        signature = parsed['signature']
        # HMAC-SHA256 produces 64-character hex string
        assert len(signature) == 64, "Signature should be 64 hex characters"
        assert all(c in '0123456789abcdef' for c in signature), \
            "Signature should be lowercase hex"

    def test_verify_signature_on_load(self, checkpoint, sample_results):
        """Verify signature is checked when loading checkpoint."""
        checkpoint.save('test_step', sample_results)

        # Tamper with the data
        checkpoint_file = checkpoint.output_dir / 'checkpoint.json'
        with open(checkpoint_file, 'r') as f:
            parsed = json.loads(f.read())

        # Modify the data but keep old signature
        data = json.loads(parsed['data'])
        data['test_step']['stations'].append('MALICIOUS_STN')
        parsed['data'] = json.dumps(data, sort_keys=True)

        with open(checkpoint_file, 'w') as f:
            json.dump(parsed, f)

        # Loading should fail or return None due to invalid signature
        result = checkpoint.load_stations()
        assert result is None, "Should reject tampered checkpoint"

    def test_invalid_signature_raises_or_returns_none(self, checkpoint, sample_results):
        """Checkpoint with invalid signature should be rejected."""
        checkpoint.save('test_step', sample_results)

        checkpoint_file = checkpoint.output_dir / 'checkpoint.json'
        with open(checkpoint_file, 'r') as f:
            parsed = json.loads(f.read())

        # Replace signature with invalid one
        parsed['signature'] = 'a' * 64

        with open(checkpoint_file, 'w') as f:
            json.dump(parsed, f)

        result = checkpoint.load_stations()
        assert result is None, "Should reject invalid signature"

    # =========================================================================
    # Test 3: Secret Key Management
    # =========================================================================

    def test_uses_environment_secret_key(self, temp_dir, sample_results):
        """Verify checkpoint uses WSE_CHECKPOINT_KEY from environment."""
        # Clear any cached key
        Checkpoint.SECRET_KEY = None

        with patch.dict(os.environ, {'WSE_CHECKPOINT_KEY': 'test-secret-key-123'}):
            checkpoint = Checkpoint(str(temp_dir))
            checkpoint.save('test_step', sample_results)

            # Create another checkpoint with same key - should load successfully
            Checkpoint.SECRET_KEY = None
            checkpoint2 = Checkpoint(str(temp_dir))
            result = checkpoint2.load_stations()
            # Result depends on what's in 'stations' key

        # Reset for other tests
        Checkpoint.SECRET_KEY = None

    def test_different_keys_produce_different_signatures(self, temp_dir, sample_results):
        """Different secret keys should produce different signatures."""
        Checkpoint.SECRET_KEY = None

        with patch.dict(os.environ, {'WSE_CHECKPOINT_KEY': 'key-one'}):
            cp1 = Checkpoint(str(temp_dir))
            cp1.save('test_step', sample_results)
            with open(cp1.output_dir / 'checkpoint.json', 'r') as f:
                sig1 = json.loads(f.read())['signature']

        # Clear and use different key
        Checkpoint.SECRET_KEY = None

        with patch.dict(os.environ, {'WSE_CHECKPOINT_KEY': 'key-two'}):
            cp2 = Checkpoint(str(temp_dir))
            cp2.save('test_step', sample_results)
            with open(cp2.output_dir / 'checkpoint.json', 'r') as f:
                sig2 = json.loads(f.read())['signature']

        assert sig1 != sig2, "Different keys should produce different signatures"

        Checkpoint.SECRET_KEY = None

    # =========================================================================
    # Test 4: Backward Compatibility (pickle migration)
    # =========================================================================

    def test_loads_legacy_pickle_with_warning(self, temp_dir, sample_results, caplog):
        """Should load old pickle files with a deprecation warning."""
        # Create a legacy pickle checkpoint
        pkl_file = temp_dir / 'checkpoint.pkl'
        legacy_data = {'stations': sample_results}
        with open(pkl_file, 'wb') as f:
            pickle.dump(legacy_data, f)

        checkpoint = Checkpoint(str(temp_dir))
        result = checkpoint.load_stations()

        # Should load the data
        assert result == sample_results, "Should load legacy pickle data"

        # Should log a warning about deprecated format
        assert any('pickle' in record.message.lower() or
                   'deprecated' in record.message.lower() or
                   'legacy' in record.message.lower()
                   for record in caplog.records), \
            "Should warn about legacy pickle format"

    def test_converts_legacy_pickle_to_json(self, temp_dir, sample_results):
        """Loading legacy pickle should convert it to signed JSON."""
        # Create legacy pickle
        pkl_file = temp_dir / 'checkpoint.pkl'
        legacy_data = {'stations': sample_results}
        with open(pkl_file, 'wb') as f:
            pickle.dump(legacy_data, f)

        checkpoint = Checkpoint(str(temp_dir))
        checkpoint.load_stations()  # Trigger migration

        # New JSON file should exist
        json_file = temp_dir / 'checkpoint.json'
        assert json_file.exists(), "Should create JSON checkpoint after migration"

        # Legacy file should be removed or renamed
        assert not pkl_file.exists() or (temp_dir / 'checkpoint.pkl.bak').exists(), \
            "Legacy pickle should be removed or backed up"

    # =========================================================================
    # Test 5: Data Integrity
    # =========================================================================

    def test_round_trip_preserves_data(self, checkpoint, sample_results):
        """Saving and loading should preserve all data."""
        checkpoint.save('stations', sample_results)

        # Create new checkpoint instance to force reload from disk
        checkpoint2 = Checkpoint(str(checkpoint.output_dir))
        loaded = checkpoint2.load_stations()

        assert loaded == sample_results, "Round-trip should preserve data"

    def test_handles_complex_data_types(self, checkpoint):
        """Should handle various data types in results."""
        complex_data = {
            'string': 'hello',
            'integer': 42,
            'float': 3.14159,
            'boolean': True,
            'null': None,
            'list': [1, 2, 3],
            'nested': {
                'a': {'b': {'c': 'deep'}},
            },
        }

        checkpoint.save('stations', complex_data)
        checkpoint2 = Checkpoint(str(checkpoint.output_dir))
        loaded = checkpoint2.load_stations()

        assert loaded == complex_data, "Should preserve complex data types"

    def test_handles_datetime_serialization(self, checkpoint):
        """Should handle datetime objects (common in checkpoint data)."""
        # Note: datetime must be serialized to string for JSON
        data_with_time = {
            'timestamp': datetime.now().isoformat(),
            'stations': ['STN001'],
        }

        checkpoint.save('stations', data_with_time)
        checkpoint2 = Checkpoint(str(checkpoint.output_dir))
        loaded = checkpoint2.load_stations()

        assert loaded == data_with_time


class TestCheckpointManagerSecurity:
    """Test CheckpointManager with secure serialization."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        return {
            'dataset': {'source': 'hydroweb'},
            'processing': {'resolution': 'glb_15min'},
            'filters': {},
        }

    def test_manager_uses_json_format(self, temp_dir, sample_config):
        """CheckpointManager should save as signed JSON."""
        manager = CheckpointManager(str(temp_dir), sample_config)
        manager.save('validate', 'completed', results={'count': 10})

        # Should create .json file, not .pkl
        json_file = temp_dir / 'step_validate.json'
        pkl_file = temp_dir / 'step_validate.pkl'

        assert json_file.exists(), "Should create .json checkpoint"
        assert not pkl_file.exists(), "Should not create .pkl checkpoint"

    def test_manager_verifies_signature_on_load(self, temp_dir, sample_config):
        """CheckpointManager should verify signatures when loading."""
        manager = CheckpointManager(str(temp_dir), sample_config)
        manager.save('validate', 'completed', results={'count': 10})

        # Tamper with file
        json_file = temp_dir / 'step_validate.json'
        with open(json_file, 'r') as f:
            parsed = json.loads(f.read())

        parsed['signature'] = 'tampered' + parsed['signature'][8:]
        with open(json_file, 'w') as f:
            json.dump(parsed, f)

        # New manager should not load tampered checkpoint
        manager2 = CheckpointManager(str(temp_dir), sample_config)
        assert manager2.load('validate') is None, \
            "Should reject tampered checkpoint"

    def test_manager_migrates_legacy_checkpoints(self, temp_dir, sample_config, caplog):
        """CheckpointManager should migrate legacy pickle checkpoints."""
        # Create legacy pickle checkpoint
        pkl_file = temp_dir / 'step_validate.pkl'
        legacy_data = CheckpointData(
            step='validate',
            status='completed',
            timestamp=datetime.now(),
            config_hash='abcd1234',
            results={'count': 10},
        )
        with open(pkl_file, 'wb') as f:
            pickle.dump(legacy_data, f)

        # Create manager - should migrate on load
        manager = CheckpointManager(str(temp_dir), sample_config)

        # Note: The migrated checkpoint may not match config_hash,
        # so it might not be loaded, but migration should occur
        json_file = temp_dir / 'step_validate.json'
        assert json_file.exists() or not pkl_file.exists(), \
            "Should attempt migration of legacy checkpoints"


class TestCheckpointEdgeCases:
    """Edge case tests for checkpoint security."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_empty_checkpoint_file(self, temp_dir):
        """Handle empty checkpoint file gracefully."""
        json_file = temp_dir / 'checkpoint.json'
        json_file.write_text('')

        checkpoint = Checkpoint(str(temp_dir))
        result = checkpoint.load_stations()
        assert result is None, "Should handle empty file"

    def test_malformed_json(self, temp_dir):
        """Handle malformed JSON gracefully."""
        json_file = temp_dir / 'checkpoint.json'
        json_file.write_text('{not valid json}')

        checkpoint = Checkpoint(str(temp_dir))
        result = checkpoint.load_stations()
        assert result is None, "Should handle malformed JSON"

    def test_missing_signature_field(self, temp_dir):
        """Reject checkpoint without signature."""
        json_file = temp_dir / 'checkpoint.json'
        json_file.write_text(json.dumps({'data': '{}'}))

        checkpoint = Checkpoint(str(temp_dir))
        result = checkpoint.load_stations()
        assert result is None, "Should reject missing signature"

    def test_missing_data_field(self, temp_dir):
        """Reject checkpoint without data."""
        json_file = temp_dir / 'checkpoint.json'
        json_file.write_text(json.dumps({'signature': 'a' * 64}))

        checkpoint = Checkpoint(str(temp_dir))
        result = checkpoint.load_stations()
        assert result is None, "Should reject missing data"

    def test_timing_attack_resistance(self, temp_dir):
        """Signature comparison should use constant-time comparison."""
        # This test verifies hmac.compare_digest is used
        # We can't easily test timing, but we can verify behavior
        checkpoint = Checkpoint(str(temp_dir))
        checkpoint.save('stations', {'test': 'data'})

        json_file = temp_dir / 'checkpoint.json'
        with open(json_file, 'r') as f:
            parsed = json.loads(f.read())

        # Try signatures that differ in different positions
        original_sig = parsed['signature']

        for i in [0, 32, 63]:  # Beginning, middle, end
            # Create signature that differs at position i
            tampered_sig = list(original_sig)
            tampered_sig[i] = 'f' if original_sig[i] != 'f' else '0'
            parsed['signature'] = ''.join(tampered_sig)

            with open(json_file, 'w') as f:
                json.dump(parsed, f)

            checkpoint2 = Checkpoint(str(temp_dir))
            result = checkpoint2.load_stations()
            assert result is None, f"Should reject signature differing at position {i}"

            # Restore for next iteration
            parsed['signature'] = original_sig
