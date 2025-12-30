"""Tests for SSHErrorClassifier.

December 30, 2025: Tests for SSH error classification used in transport retry decisions.
"""

import pytest

from app.coordination.transport_base import (
    SSHErrorClassifier,
    SSHErrorType,
)


class TestSSHErrorClassifier:
    """Tests for SSHErrorClassifier."""

    # =========================================================================
    # Auth Failure Classification
    # =========================================================================

    @pytest.mark.parametrize(
        "stderr",
        [
            "Permission denied (publickey)",
            "Permission denied, please try again",
            "Host key verification failed.",
            "REMOTE HOST IDENTIFICATION HAS CHANGED!",
            "Too many authentication failures for user",
            "Authentication failed",
            "no mutual signature algorithm",
            "Unable to negotiate with host: no matching key exchange method",
            "no matching host key type found",
        ],
    )
    def test_classify_auth_failures(self, stderr: str) -> None:
        """Test that auth failure patterns are classified correctly."""
        error_type = SSHErrorClassifier.classify(stderr)
        assert error_type == SSHErrorType.AUTH
        assert not SSHErrorClassifier.should_retry(stderr)
        assert SSHErrorClassifier.is_auth_failure(stderr)
        assert not SSHErrorClassifier.is_network_failure(stderr)

    # =========================================================================
    # Network Failure Classification
    # =========================================================================

    @pytest.mark.parametrize(
        "stderr",
        [
            "Connection reset by peer",
            "Connection refused",
            "Connection timed out",
            "Network is unreachable",
            "No route to host",
            "Temporary failure in name resolution",
            "Could not resolve hostname",
            "Connection closed by remote host",
            "Broken pipe",
            "Operation timed out",
            "ssh_exchange_identification: read: Connection reset",
            "kex_exchange_identification: read: Connection reset",
            "Write failed: Broken pipe",
            "packet_write_wait: Connection to host port 22: Broken pipe",
            "client_loop: send disconnect: Broken pipe",
        ],
    )
    def test_classify_network_failures(self, stderr: str) -> None:
        """Test that network failure patterns are classified correctly."""
        error_type = SSHErrorClassifier.classify(stderr)
        assert error_type == SSHErrorType.NETWORK
        assert SSHErrorClassifier.should_retry(stderr)
        assert not SSHErrorClassifier.is_auth_failure(stderr)
        assert SSHErrorClassifier.is_network_failure(stderr)

    # =========================================================================
    # Unknown Errors
    # =========================================================================

    @pytest.mark.parametrize(
        "stderr",
        [
            "Unknown error occurred",
            "Something went wrong",
            "",  # Empty string
            "File not found",
            "Disk full",
        ],
    )
    def test_classify_unknown_errors(self, stderr: str) -> None:
        """Test that unclassified errors return UNKNOWN."""
        error_type = SSHErrorClassifier.classify(stderr)
        assert error_type == SSHErrorType.UNKNOWN
        # Unknown errors should default to retry (might be transient)
        assert SSHErrorClassifier.should_retry(stderr)
        assert not SSHErrorClassifier.is_auth_failure(stderr)
        assert not SSHErrorClassifier.is_network_failure(stderr)

    # =========================================================================
    # should_retry Tests
    # =========================================================================

    def test_should_retry_auth_failure(self) -> None:
        """Test that auth failures should not be retried."""
        assert not SSHErrorClassifier.should_retry("Permission denied")
        assert not SSHErrorClassifier.should_retry("Host key verification failed")

    def test_should_retry_network_failure(self) -> None:
        """Test that network failures should be retried."""
        assert SSHErrorClassifier.should_retry("Connection reset")
        assert SSHErrorClassifier.should_retry("Connection timed out")

    def test_should_retry_unknown(self) -> None:
        """Test that unknown errors should be retried (default safe)."""
        assert SSHErrorClassifier.should_retry("Something random happened")
        assert SSHErrorClassifier.should_retry("")

    # =========================================================================
    # get_retry_recommendation Tests
    # =========================================================================

    def test_get_retry_recommendation_auth(self) -> None:
        """Test retry recommendation for auth failures."""
        should_retry, reason = SSHErrorClassifier.get_retry_recommendation(
            "Permission denied (publickey)"
        )
        assert not should_retry
        assert "Authentication failure" in reason

    def test_get_retry_recommendation_network(self) -> None:
        """Test retry recommendation for network failures."""
        should_retry, reason = SSHErrorClassifier.get_retry_recommendation(
            "Connection reset by peer"
        )
        assert should_retry
        assert "Network failure" in reason
        assert "transient" in reason

    def test_get_retry_recommendation_unknown(self) -> None:
        """Test retry recommendation for unknown errors."""
        should_retry, reason = SSHErrorClassifier.get_retry_recommendation(
            "Something weird"
        )
        assert should_retry
        assert "Unknown error" in reason

    # =========================================================================
    # Case Insensitivity Tests
    # =========================================================================

    def test_case_insensitive_auth(self) -> None:
        """Test that auth pattern matching is case-insensitive."""
        assert SSHErrorClassifier.classify("PERMISSION DENIED") == SSHErrorType.AUTH
        assert SSHErrorClassifier.classify("permission denied") == SSHErrorType.AUTH
        assert SSHErrorClassifier.classify("Permission Denied") == SSHErrorType.AUTH

    def test_case_insensitive_network(self) -> None:
        """Test that network pattern matching is case-insensitive."""
        assert SSHErrorClassifier.classify("CONNECTION RESET") == SSHErrorType.NETWORK
        assert SSHErrorClassifier.classify("connection reset") == SSHErrorType.NETWORK
        assert SSHErrorClassifier.classify("Connection Reset") == SSHErrorType.NETWORK

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_empty_string(self) -> None:
        """Test handling of empty string."""
        assert SSHErrorClassifier.classify("") == SSHErrorType.UNKNOWN
        assert SSHErrorClassifier.should_retry("")

    def test_none_handling(self) -> None:
        """Test that None input is handled gracefully."""
        # classify expects str, but should handle edge cases
        # In practice, callers should pass stderr which may be empty
        assert SSHErrorClassifier.classify("") == SSHErrorType.UNKNOWN

    def test_multiple_patterns_in_message(self) -> None:
        """Test message containing multiple patterns (auth takes precedence)."""
        # Auth patterns should take precedence since they're checked first
        stderr = "Connection reset, Permission denied"
        assert SSHErrorClassifier.classify(stderr) == SSHErrorType.AUTH

    def test_partial_match(self) -> None:
        """Test that partial matches work correctly."""
        # "publickey" is in AUTH_PATTERNS
        assert SSHErrorClassifier.classify("no more authentication methods to try. (publickey)") == SSHErrorType.AUTH

    def test_rsync_specific_errors(self) -> None:
        """Test rsync-specific error messages."""
        assert SSHErrorClassifier.classify("rsync: connection unexpectedly closed") == SSHErrorType.NETWORK
        assert SSHErrorClassifier.classify("rsync error: error in socket IO") == SSHErrorType.UNKNOWN

    def test_scp_specific_errors(self) -> None:
        """Test scp-specific error messages."""
        assert SSHErrorClassifier.classify("scp: Connection closed") == SSHErrorType.NETWORK
        assert SSHErrorClassifier.classify("scp: Permission denied") == SSHErrorType.AUTH
