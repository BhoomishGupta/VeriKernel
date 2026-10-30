#!/usr/bin/env python3
"""
Unit and Integration Tests for VeriKernel API Guards & Availability Protection.
Tests:
  1. /api/health and /api/budget-status telemetry
  2. IP-based hourly rate limiting (5 calls/hour per IP -> 6th triggers HTTP 429)
  3. Global daily execution cap (30 live runs/day -> HTTP 429)
  4. Daily cloud spending budget limit ($5.00 limit -> HTTP 429)
  5. Cached referee runs (0-token gallery calls) remain unaffected by rate limiter
"""

import pathlib
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

import server
from server import app


class TestVeriKernelApiGuards(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Reset IP call history between tests
        server.ip_call_history.clear()

    def test_health_endpoint(self):
        """Verify health check returns status ok and availability guard metadata."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("guards", data)
        self.assertEqual(data["guards"]["ip_hourly_limit"], 5)
        self.assertEqual(data["guards"]["daily_runs_limit"], 30)
        self.assertEqual(data["guards"]["daily_spend_limit_usd"], 5.00)

    def test_budget_status_endpoint(self):
        """Verify budget status reports correct provider metadata and quotas."""
        resp = self.client.get("/api/budget-status", headers={"X-Forwarded-For": "198.51.100.1"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["daily_runs_limit"], 30)
        self.assertEqual(data["ip_hourly_limit"], 5)
        self.assertEqual(data["client_ip"], "198.51.100.1")
        self.assertEqual(data["ip_calls_used"], 0)
        self.assertEqual(data["ip_calls_remaining"], 5)
        self.assertIn("daily_spend_remaining_usd", data)

    def test_cached_gallery_not_rate_limited(self):
        """Cached Break-It Gallery requests (0 LLM tokens, in-memory) must bypass rate limits."""
        for _ in range(8):
            resp = self.client.post(
                "/api/run-referee",
                json={"gallery_bug": "gallery/bug_1_fast_mathcheat.hpp", "force_live": False},
                headers={"X-Forwarded-For": "203.0.113.10"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data.get("status"), "success")

    @patch("server.run_referee_sync")
    def test_ip_rate_limiting(self, mock_referee):
        """Live executions must enforce a strict cap of 5 calls per hour per IP."""
        mock_referee.return_value = [{"stage": "Stage 1", "ok": True, "status_tag": "PASS"}]
        test_ip = "192.0.2.42"

        # First 5 live referee executions should be allowed
        for i in range(5):
            resp = self.client.post(
                "/api/run-referee",
                json={"target": "baseline", "force_live": True},
                headers={"X-Forwarded-For": test_ip},
            )
            # Response must succeed (200)
            self.assertEqual(resp.status_code, 200, f"Call {i+1} failed unexpectedly")

        # 6th call from same IP must be intercepted with HTTP 429
        resp_blocked = self.client.post(
            "/api/run-referee",
            json={"target": "baseline", "force_live": True},
            headers={"X-Forwarded-For": test_ip},
        )
        self.assertEqual(resp_blocked.status_code, 429)
        self.assertIn("Rate limit exceeded", resp_blocked.json()["detail"])
        self.assertIn("5 live calls per hour", resp_blocked.json()["detail"])

        # Different IP must still have its full quota available
        different_ip = "192.0.2.99"
        resp_diff = self.client.post(
            "/api/run-referee",
            json={"target": "baseline", "force_live": True},
            headers={"X-Forwarded-For": different_ip},
        )
        self.assertEqual(resp_diff.status_code, 200)

    def test_global_daily_cap(self):
        """Global execution cap must intercept requests when daily limit is reached."""
        orig_limit = server.DAILY_RUNS_LIMIT
        try:
            # Artificially set limit to 0
            server.DAILY_RUNS_LIMIT = 0
            resp = self.client.post(
                "/api/run-referee",
                json={"target": "baseline", "force_live": True},
                headers={"X-Forwarded-For": "198.51.100.50"},
            )
            self.assertEqual(resp.status_code, 429)
            self.assertIn("Global daily execution cap reached", resp.json()["detail"])
        finally:
            server.DAILY_RUNS_LIMIT = orig_limit

    def test_daily_spending_limit(self):
        """Cloud spending guard must halt live LLM runs when budget is exceeded."""
        orig_limit = server.DAILY_SPEND_LIMIT_USD
        try:
            # Artificially set limit to $0.00
            server.DAILY_SPEND_LIMIT_USD = 0.00
            resp = self.client.post(
                "/api/run-agent-live",
                headers={"X-Forwarded-For": "198.51.100.60"},
            )
            self.assertEqual(resp.status_code, 429)
            self.assertIn("Daily cloud spending budget reached", resp.json()["detail"])
        finally:
            server.DAILY_SPEND_LIMIT_USD = orig_limit

    @patch("server.run_referee_sync")
    def test_judge_token_bypasses_ip_rate_limit(self, mock_referee):
        """Verified judges carrying X-Judge-Token bypass the 5 calls/hour IP rate limit."""
        mock_referee.return_value = [{"stage": "Stage 1", "ok": True, "status_tag": "PASS"}]
        test_ip = "192.0.2.77"

        # Execute 8 consecutive live calls with Judge Token (normally capped at 5)
        for i in range(8):
            resp = self.client.post(
                "/api/run-referee",
                json={"target": "baseline", "force_live": True},
                headers={
                    "X-Forwarded-For": test_ip,
                    "X-Judge-Token": "verikernel-judge-2026",
                },
            )
            self.assertEqual(resp.status_code, 200, f"Judge call {i+1} was unexpectedly throttled")

    @patch("server.run_referee_sync")
    def test_judge_token_bypasses_daily_run_cap(self, mock_referee):
        """Judges carrying X-Judge-Token bypass the daily execution cap to evaluate kernels."""
        mock_referee.return_value = [{"stage": "Stage 1", "ok": True, "status_tag": "PASS"}]
        orig_limit = server.DAILY_RUNS_LIMIT
        try:
            server.DAILY_RUNS_LIMIT = 0
            # Unauthenticated call should fail
            resp_blocked = self.client.post(
                "/api/run-referee",
                json={"target": "baseline", "force_live": True},
                headers={"X-Forwarded-For": "198.51.100.80"},
            )
            self.assertEqual(resp_blocked.status_code, 429)

            # Authenticated Judge call should succeed
            resp_judge = self.client.post(
                "/api/run-referee",
                json={"target": "baseline", "force_live": True},
                headers={
                    "X-Forwarded-For": "198.51.100.80",
                    "X-Judge-Token": "verikernel-judge-2026",
                },
            )
            self.assertEqual(resp_judge.status_code, 200)

            # Query param judge_token should also succeed
            resp_query = self.client.post(
                "/api/run-referee?judge_token=verikernel-judge-2026",
                json={"target": "baseline", "force_live": True},
                headers={"X-Forwarded-For": "198.51.100.80"},
            )
            self.assertEqual(resp_query.status_code, 200)
        finally:
            server.DAILY_RUNS_LIMIT = orig_limit

    def test_judge_still_respects_financial_spend_limit(self):
        """Even with a Judge token, the financial spend limit protects cloud credits from overdraft."""
        orig_limit = server.DAILY_SPEND_LIMIT_USD
        try:
            server.DAILY_SPEND_LIMIT_USD = 0.00
            resp = self.client.post(
                "/api/run-agent-live",
                headers={
                    "X-Forwarded-For": "198.51.100.90",
                    "X-Judge-Token": "verikernel-judge-2026",
                },
            )
            self.assertEqual(resp.status_code, 429)
            self.assertIn("Daily cloud spending budget reached", resp.json()["detail"])
        finally:
            server.DAILY_SPEND_LIMIT_USD = orig_limit

    def test_budget_status_reports_judge_status(self):
        """Verify /api/budget-status accurately reflects judge VIP authentication."""
        # Unauthenticated
        resp1 = self.client.get("/api/budget-status")
        self.assertEqual(resp1.status_code, 200)
        self.assertFalse(resp1.json()["is_judge"])

        # Authenticated with valid token
        resp2 = self.client.get("/api/budget-status", headers={"X-Judge-Token": "verikernel-judge-2026"})
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(resp2.json()["is_judge"])
        self.assertEqual(resp2.json()["ip_calls_remaining"], server.JUDGE_HOURLY_LIMIT)

    @patch("server.run_referee_sync")
    def test_judge_rate_limit_enforced(self, mock_referee):
        """Judge VIP passkey has a protective rate limit (20 calls/hr) preventing bot abuse."""
        mock_referee.return_value = [{"stage": "Stage 1", "ok": True, "status_tag": "PASS"}]
        test_ip = "192.0.2.88"
        server.judge_call_history[test_ip] = []

        # Reach the limit
        for _ in range(server.JUDGE_HOURLY_LIMIT):
            resp = self.client.post(
                "/api/run-referee",
                json={"target": "baseline", "force_live": True},
                headers={
                    "X-Forwarded-For": test_ip,
                    "X-Judge-Token": "verikernel-judge-2026",
                },
            )
            self.assertEqual(resp.status_code, 200)

        # Next call should be rate limited
        resp_blocked = self.client.post(
            "/api/run-referee",
            json={"target": "baseline", "force_live": True},
            headers={
                "X-Forwarded-For": test_ip,
                "X-Judge-Token": "verikernel-judge-2026",
            },
        )
        self.assertEqual(resp_blocked.status_code, 429)
        self.assertIn("Judge VIP rate limit reached", resp_blocked.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
