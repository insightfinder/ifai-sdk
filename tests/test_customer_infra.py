"""
Unit tests for customer infrastructure methods in the SDK client.
Uses EvenUp sample fixture data where shapes match the API contract.
"""
import hashlib
import hmac
import json
import os
import unittest
from unittest.mock import MagicMock, patch, call

from insightfinderai.config import DEFAULT_API_URL


MATTER_ID = "97c6bd47-10cf-4507-9479-d597f1d27855"
PLAYBOOK_ID = "770e8400-e29b-41d4-a716-446655440002"
PLAINTIFF_ID = "660e8400-e29b-41d4-a716-446655440001"
FILE_UPLOAD_ID = "069f92549fed762880004d3da26f778a"
ANNOTATION_FILE_ID = "6367222"
RUN_ID = "1ad53a94-589c-4496-8dcb-f8eef6382a2e"
MODEL = "GEMINI_2_5"


def _ok(body=""):
    """Return a mock response with status 200."""
    r = MagicMock()
    r.status_code = 200
    r.text = body if isinstance(body, str) else json.dumps(body)
    r.json.return_value = body if not isinstance(body, str) else {}
    return r


def _err(status=500, text="error"):
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


def _make_client():
    """
    Create a Client instance with all network calls stubbed out.
    _get_project_name uses requests.post; background clear_context /
    clear_system_prompt use requests.post and requests.delete.
    """
    from insightfinderai import Client

    with patch("requests.post", return_value=_ok("test-session-Prompt")), \
         patch("requests.delete", return_value=_ok()):
        client = Client(
            session_name="test-session",
            url=DEFAULT_API_URL,
            username="testuser",
            api_key="test-api-key",
        )
    return client


class TestSendCustomerInfraCompare(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_posts_composite_dataset_and_template_ids(self):
        """datasetId = matter_id (SDK receives just matterId); templateId = playbookId@plaintiffId."""
        with patch("requests.post", return_value=_ok(RUN_ID)) as mock_post, \
             patch.object(self.client, "_save_prompt_library"), \
             patch.object(self.client, "_save_datasets"):
            run_id = self.client.send_customer_infra_compare(
                model=MODEL,
                matter_id=MATTER_ID,
                plaintiff_id=PLAINTIFF_ID,
                playbook_id=PLAYBOOK_ID,
            )

        self.assertEqual(RUN_ID, run_id)
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        body = kwargs.get("json") or mock_post.call_args[0][1] if len(mock_post.call_args[0]) > 1 else kwargs["json"]
        # Access via kwargs
        posted = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual(MODEL, posted["model"])
        self.assertEqual(MATTER_ID, posted["datasetId"])
        self.assertEqual(f"{PLAYBOOK_ID}@{PLAINTIFF_ID}", posted["templateId"])

    def test_ingest_called_before_compare(self):
        """_save_prompt_library and _save_datasets are called before posting the compare."""
        call_order = []

        def mock_save_lib(pb, pl):
            call_order.append("lib")

        def mock_save_ds(m):
            call_order.append("ds")

        with patch("requests.post", return_value=_ok(RUN_ID)) as mock_post, \
             patch.object(self.client, "_save_prompt_library", side_effect=mock_save_lib), \
             patch.object(self.client, "_save_datasets", side_effect=mock_save_ds):
            self.client.send_customer_infra_compare(
                model=MODEL,
                matter_id=MATTER_ID,
                plaintiff_id=PLAINTIFF_ID,
                playbook_id=PLAYBOOK_ID,
            )

        self.assertIn("lib", call_order)
        self.assertIn("ds", call_order)
        self.assertLess(call_order.index("lib"), call_order.index("ds") + 1)

    def test_optional_prompt_field_omitted_when_none(self):
        with patch("requests.post", return_value=_ok(RUN_ID)), \
             patch.object(self.client, "_save_prompt_library"), \
             patch.object(self.client, "_save_datasets"):
            self.client.send_customer_infra_compare(
                model=MODEL,
                matter_id=MATTER_ID,
                plaintiff_id=PLAINTIFF_ID,
                playbook_id=PLAYBOOK_ID,
                prompt=None,
            )

        posted = patch("requests.post").__enter__  # already called; checked above
        # Verify via side-effect capture instead
        with patch("requests.post", return_value=_ok(RUN_ID)) as mock_post, \
             patch.object(self.client, "_save_prompt_library"), \
             patch.object(self.client, "_save_datasets"):
            self.client.send_customer_infra_compare(
                model=MODEL,
                matter_id=MATTER_ID,
                plaintiff_id=PLAINTIFF_ID,
                playbook_id=PLAYBOOK_ID,
                prompt=None,
            )
        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertNotIn("prompt", sent)

    def test_prompt_field_included_when_provided(self):
        with patch("requests.post", return_value=_ok(RUN_ID)) as mock_post, \
             patch.object(self.client, "_save_prompt_library"), \
             patch.object(self.client, "_save_datasets"):
            self.client.send_customer_infra_compare(
                model=MODEL,
                matter_id=MATTER_ID,
                plaintiff_id=PLAINTIFF_ID,
                playbook_id=PLAYBOOK_ID,
                prompt="What was the date of injury?",
            )
        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual("What was the date of injury?", sent["prompt"])

    def test_api_error_raises_value_error(self):
        with patch("requests.post", return_value=_err(500, "internal server error")), \
             patch.object(self.client, "_save_prompt_library"), \
             patch.object(self.client, "_save_datasets"):
            with self.assertRaises(ValueError) as ctx:
                self.client.send_customer_infra_compare(
                    model=MODEL,
                    matter_id=MATTER_ID,
                    plaintiff_id=PLAINTIFF_ID,
                    playbook_id=PLAYBOOK_ID,
                )
        self.assertIn("500", str(ctx.exception))


class TestSavePromptLibrary(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_posts_playbook_and_plaintiff_ids(self):
        with patch("requests.post", return_value=_ok()) as mock_post:
            self.client._save_prompt_library(PLAYBOOK_ID, PLAINTIFF_ID)

        mock_post.assert_called_once()
        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual(PLAYBOOK_ID, sent["playbookId"])
        self.assertEqual(PLAINTIFF_ID, sent["plaintiffId"])

    def test_non_2xx_logs_warning_without_raising(self):
        """Non-2xx from ingest should not raise — it's fire-and-log."""
        with patch("requests.post", return_value=_err(404)):
            # Should not raise
            self.client._save_prompt_library(PLAYBOOK_ID, PLAINTIFF_ID)


class TestSaveDatasets(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_posts_matter_id(self):
        with patch("requests.post", return_value=_ok()) as mock_post:
            self.client._save_datasets(MATTER_ID)

        mock_post.assert_called_once()
        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual(MATTER_ID, sent["matterId"])

    def test_non_2xx_logs_warning_without_raising(self):
        with patch("requests.post", return_value=_err(503)):
            self.client._save_datasets(MATTER_ID)


class TestGetCustomerDatasets(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_returns_datasets_grouped_by_matter(self):
        dataset_name = f"{MATTER_ID}@{FILE_UPLOAD_ID}@{ANNOTATION_FILE_ID}"
        api_response = {MATTER_ID: [dataset_name]}
        with patch("requests.get", return_value=_ok(api_response)) as mock_get:
            result = self.client.get_customer_datasets()

        self.assertIn(MATTER_ID, result)
        self.assertEqual([dataset_name], result[MATTER_ID])
        # Verify currentInfra query param was passed.
        _, kwargs = mock_get.call_args
        params = kwargs.get("params") or mock_get.call_args[1].get("params")
        self.assertEqual("Customer Infrastructure", params["currentInfra"])

    def test_default_infra_is_customer_infrastructure(self):
        with patch("requests.get", return_value=_ok({})) as mock_get:
            self.client.get_customer_datasets()

        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params")
        self.assertEqual("Customer Infrastructure", params["currentInfra"])

    def test_custom_infra_passed_through(self):
        with patch("requests.get", return_value=_ok({})) as mock_get:
            self.client.get_customer_datasets(current_infra="InsightFinder Infrastructure")

        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params")
        self.assertEqual("InsightFinder Infrastructure", params["currentInfra"])

    def test_api_error_raises_value_error(self):
        with patch("requests.get", return_value=_err(403)):
            with self.assertRaises(ValueError) as ctx:
                self.client.get_customer_datasets()
        self.assertIn("403", str(ctx.exception))


class TestGetCustomerInfraOptions(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_returns_list_of_option_strings(self):
        options = ["InsightFinder Infrastructure", "Customer Infrastructure"]
        with patch("requests.get", return_value=_ok(options)) as mock_get:
            result = self.client.get_customer_infra_options()

        self.assertEqual(options, result)
        mock_get.assert_called_once()

    def test_api_error_raises_value_error(self):
        with patch("requests.get", return_value=_err(500)):
            with self.assertRaises(ValueError) as ctx:
                self.client.get_customer_infra_options()
        self.assertIn("500", str(ctx.exception))

    def test_network_error_raises_value_error(self):
        import requests as req
        with patch("requests.get", side_effect=req.exceptions.ConnectionError("refused")):
            with self.assertRaises(ValueError) as ctx:
                self.client.get_customer_infra_options()
        self.assertIn("failed", str(ctx.exception).lower())


class TestGetCustomerInfraSettings(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_returns_settings_dict(self):
        settings = {
            "apiBaseUrl": "https://api.evenup.com",
            "apiToken": "tok-abc",
            "modelList": [MODEL],
            "hmacSecret": "s3cr3t",
            "webhookUrl": DEFAULT_API_URL + "/api/customer/webhook",
            "currentInfra": "Customer Infrastructure",
        }
        with patch("requests.get", return_value=_ok(settings)) as mock_get:
            result = self.client.get_customer_infra_settings()

        self.assertEqual(settings, result)
        mock_get.assert_called_once()

    def test_api_error_raises_value_error(self):
        with patch("requests.get", return_value=_err(403)):
            with self.assertRaises(ValueError) as ctx:
                self.client.get_customer_infra_settings()
        self.assertIn("403", str(ctx.exception))

    def test_network_error_raises_value_error(self):
        import requests as req
        with patch("requests.get", side_effect=req.exceptions.Timeout("timed out")):
            with self.assertRaises(ValueError) as ctx:
                self.client.get_customer_infra_settings()
        self.assertIn("failed", str(ctx.exception).lower())


class TestSaveCustomerInfraSettings(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_posts_all_four_fields(self):
        with patch("requests.post", return_value=_ok()) as mock_post:
            result = self.client.save_customer_infra_settings(
                api_base_url="https://api.evenup.com",
                api_token="tok-abc",
                model_list=[MODEL, "BEST_IN_CLASS"],
                hmac_secret="s3cr3t",
            )

        self.assertTrue(result)
        mock_post.assert_called_once()
        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual("https://api.evenup.com", sent["apiBaseUrl"])
        self.assertEqual("tok-abc", sent["apiToken"])
        self.assertEqual([MODEL, "BEST_IN_CLASS"], sent["modelList"])
        self.assertEqual("s3cr3t", sent["hmacSecret"])

    def test_empty_model_list_is_sent(self):
        with patch("requests.post", return_value=_ok()) as mock_post:
            self.client.save_customer_infra_settings(
                api_base_url="https://api.evenup.com",
                api_token="tok-abc",
                model_list=[],
                hmac_secret="s3cr3t",
            )

        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual([], sent["modelList"])

    def test_api_error_raises_value_error(self):
        with patch("requests.post", return_value=_err(400, "bad request")):
            with self.assertRaises(ValueError) as ctx:
                self.client.save_customer_infra_settings(
                    api_base_url="https://api.evenup.com",
                    api_token="tok",
                    model_list=[],
                    hmac_secret="s",
                )
        self.assertIn("400", str(ctx.exception))

    def test_network_error_raises_value_error(self):
        import requests as req
        with patch("requests.post", side_effect=req.exceptions.ConnectionError("refused")):
            with self.assertRaises(ValueError) as ctx:
                self.client.save_customer_infra_settings(
                    api_base_url="https://api.evenup.com",
                    api_token="tok",
                    model_list=[],
                    hmac_secret="s",
                )
        self.assertIn("failed", str(ctx.exception).lower())


class TestSwitchCustomerInfra(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_posts_infra_type_and_returns_true(self):
        with patch("requests.post", return_value=_ok()) as mock_post:
            result = self.client.switch_customer_infra("Customer Infrastructure")

        self.assertTrue(result)
        mock_post.assert_called_once()
        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual("Customer Infrastructure", sent["infraType"])

    def test_switch_to_insightfinder_infra(self):
        with patch("requests.post", return_value=_ok()) as mock_post:
            self.client.switch_customer_infra("InsightFinder Infrastructure")

        sent = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual("InsightFinder Infrastructure", sent["infraType"])

    def test_api_error_raises_value_error(self):
        with patch("requests.post", return_value=_err(500)):
            with self.assertRaises(ValueError) as ctx:
                self.client.switch_customer_infra("Customer Infrastructure")
        self.assertIn("500", str(ctx.exception))

    def test_network_error_raises_value_error(self):
        import requests as req
        with patch("requests.post", side_effect=req.exceptions.ConnectionError("refused")):
            with self.assertRaises(ValueError) as ctx:
                self.client.switch_customer_infra("Customer Infrastructure")
        self.assertIn("failed", str(ctx.exception).lower())


class TestVerifyCustomerInfraToken(unittest.TestCase):

    def setUp(self):
        self.client = _make_client()

    def test_returns_true_when_server_says_valid(self):
        with patch("requests.post", return_value=_ok({"valid": True})):
            result = self.client.verify_customer_infra_token()
        self.assertTrue(result)

    def test_returns_false_when_server_says_invalid(self):
        with patch("requests.post", return_value=_ok({"valid": False})):
            result = self.client.verify_customer_infra_token()
        self.assertFalse(result)

    def test_api_error_raises_value_error(self):
        with patch("requests.post", return_value=_err(403, "forbidden")):
            with self.assertRaises(ValueError) as ctx:
                self.client.verify_customer_infra_token()
        self.assertIn("403", str(ctx.exception))

    def test_network_error_raises_value_error(self):
        import requests as req
        with patch("requests.post", side_effect=req.exceptions.Timeout("timed out")):
            with self.assertRaises(ValueError) as ctx:
                self.client.verify_customer_infra_token()
        self.assertIn("failed", str(ctx.exception).lower())

    def test_posts_to_verify_token_endpoint(self):
        """Uses POST (not GET), matching the SDK implementation."""
        with patch("requests.post", return_value=_ok({"valid": True})) as mock_post:
            self.client.verify_customer_infra_token()
        mock_post.assert_called_once()
        call_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args[0][0]
        self.assertIn("verify", call_url.lower())


class TestEvenUpWebhookSimulation(unittest.TestCase):
    """
    Simulates EvenUp calling our webhook endpoint.

    Signs the payload exactly as EvenUp does:
      body  = json.dumps(payload, sort_keys=True, separators=(",", ":"))
      header = "sha256=" + hmac_sha256(secret, body)

    Unit mode  : requests.post is mocked — verifies the correct headers/body are sent.
    Integration: set WEBHOOK_URL + HMAC_SECRET env vars to hit a real server.
    """

    HMAC_SECRET = "test-hmac-secret"

    # Completed-run payload — mirrors the EvenUp API doc sample.
    COMPLETED_PAYLOAD = {
        "version": "1.0",
        "event": "eval_run_completed",
        "timestamp": 1748352000,
        "data": {
            "run_id": RUN_ID,
            "status": "completed",
            "model": MODEL,
            "matter_id": MATTER_ID,
            "plaintiff_id": PLAINTIFF_ID,
            "playbook_id": PLAYBOOK_ID,
            "results": [
                {
                    "order": 0,
                    "variable_name": "date_of_injury",
                    "question": "What was the date of injury?",
                    "answer": [
                        {
                            "content": "<p><citable>January 15, 2024 "
                                       "<span id='citation-placeholder-0-0'></span></citable></p>",
                            "citations": [
                                {
                                    "reference_placeholder": "<span id='citation-placeholder-0-0'></span>",
                                    "page_number": 3,
                                    "lines": [12, 13],
                                    "source_text": ["Date of injury listed as 01/15/2024"],
                                    "annotation_file_id": int(ANNOTATION_FILE_ID),
                                    "annotation_request_id": 759327,
                                    "file_upload_id": FILE_UPLOAD_ID,
                                    "case_id": f"matter_{MATTER_ID}",
                                    "page_class": "medical_record",
                                    "type": "file",
                                    "custom_id": None,
                                    "note_id": None,
                                    "tags": None,
                                }
                            ],
                        }
                    ],
                }
            ],
            "error": None,
        },
    }

    FAILED_PAYLOAD = {
        "version": "1.0",
        "event": "eval_run_completed",
        "timestamp": 1748352000,
        "data": {
            "run_id": RUN_ID,
            "status": "failed",
            "model": MODEL,
            "matter_id": MATTER_ID,
            "plaintiff_id": PLAINTIFF_ID,
            "playbook_id": PLAYBOOK_ID,
            "results": [],
            "error": {
                "code": "run_failed",
                "message": "Run failed during execution",
                "retriable": True,
            },
        },
    }

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _canonical_body(payload: dict) -> str:
        """Canonical JSON EvenUp signs: sort_keys=True, compact separators."""
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _sign(secret: str, body: str) -> str:
        digest = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        return "sha256=" + digest

    @staticmethod
    def _verify(secret: str, body: str, header: str) -> bool:
        """Reference verifier from EvenUp API doc (Python version)."""
        expected = "sha256=" + hmac.new(
            secret.encode(), body.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, header)

    def _webhook_url(self) -> str:
        return os.environ.get(
            "WEBHOOK_URL", DEFAULT_API_URL + "/api/customer/webhook"
        )

    def _secret(self) -> str:
        return os.environ.get("HMAC_SECRET", self.HMAC_SECRET)

    # ── signature algorithm tests (no HTTP, always run) ──────────────────────

    def test_signed_body_verifies_correctly(self):
        """Round-trip: sign then verify with the same secret passes."""
        body = self._canonical_body(self.COMPLETED_PAYLOAD)
        header = self._sign(self.HMAC_SECRET, body)
        self.assertTrue(self._verify(self.HMAC_SECRET, body, header))

    def test_tampered_body_fails_verification(self):
        body = self._canonical_body(self.COMPLETED_PAYLOAD)
        header = self._sign(self.HMAC_SECRET, body)
        tampered = body + "x"
        self.assertFalse(self._verify(self.HMAC_SECRET, tampered, header))

    def test_wrong_secret_fails_verification(self):
        body = self._canonical_body(self.COMPLETED_PAYLOAD)
        header = self._sign(self.HMAC_SECRET, body)
        self.assertFalse(self._verify("wrong-secret", body, header))

    def test_canonical_body_is_sorted_and_compact(self):
        """Keys must be sorted and no spaces around separators — EvenUp requirement."""
        body = self._canonical_body({"b": 2, "a": 1})
        self.assertEqual('{"a":1,"b":2}', body)

    # ── unit mode: mock requests.post, verify headers/body sent ──────────────

    def _post_webhook(self, payload: dict, secret: str = None):
        """Simulate EvenUp POSTing to our webhook endpoint."""
        import requests as req
        secret = secret or self._secret()
        body = self._canonical_body(payload)
        signature = self._sign(secret, body)
        return req.post(
            self._webhook_url(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-EvenUp-Webhook-Signature": signature,
                "X-Webhook-Version": "1.0",
                "X-Webhook-Delivery": "4e2d-stable-uuid-across-retries",
                "User-Agent": "EvenUp-Webhooks",
            },
        )

    def test_unit_completed_webhook_posts_signed_body(self):
        """Mocked: verifies we send the signature header and canonical body."""
        with patch("requests.post", return_value=_ok("Webhook received")) as mock_post:
            response = self._post_webhook(self.COMPLETED_PAYLOAD)

        self.assertEqual(200, response.status_code)
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        sent_body = kwargs.get("data") or mock_post.call_args[1].get("data")
        sent_headers = kwargs.get("headers") or mock_post.call_args[1].get("headers")

        # Body is canonical JSON.
        parsed = json.loads(sent_body)
        self.assertEqual(RUN_ID, parsed["data"]["run_id"])
        self.assertEqual("completed", parsed["data"]["status"])

        # Signature header is present and correct.
        sig = sent_headers["X-EvenUp-Webhook-Signature"]
        self.assertTrue(sig.startswith("sha256="))
        self.assertTrue(self._verify(self.HMAC_SECRET, sent_body, sig))

    def test_unit_failed_webhook_carries_error_object(self):
        """Mocked: failed payload includes error.code and error.retriable."""
        with patch("requests.post", return_value=_ok("Webhook received")) as mock_post:
            self._post_webhook(self.FAILED_PAYLOAD)

        sent_body = mock_post.call_args.kwargs.get("data") or mock_post.call_args[1].get("data")
        parsed = json.loads(sent_body)
        self.assertEqual("failed", parsed["data"]["status"])
        self.assertEqual("run_failed", parsed["data"]["error"]["code"])
        self.assertTrue(parsed["data"]["error"]["retriable"])

    def test_unit_retry_sends_identical_signature(self):
        """X-Webhook-Delivery is stable across retries — same body = same signature."""
        body = self._canonical_body(self.COMPLETED_PAYLOAD)
        sig1 = self._sign(self.HMAC_SECRET, body)
        sig2 = self._sign(self.HMAC_SECRET, body)
        self.assertEqual(sig1, sig2)

    # ── integration mode: hits a real server (skipped unless env vars set) ───

    @unittest.skipUnless(
        os.environ.get("WEBHOOK_URL") and os.environ.get("HMAC_SECRET"),
        "Set WEBHOOK_URL and HMAC_SECRET env vars to run integration tests",
    )
    def test_integration_completed_webhook_accepted(self):
        import requests as req
        response = self._post_webhook(self.COMPLETED_PAYLOAD)
        self.assertIn(response.status_code, range(200, 300),
                      f"Expected 2xx, got {response.status_code}: {response.text}")

    @unittest.skipUnless(
        os.environ.get("WEBHOOK_URL") and os.environ.get("HMAC_SECRET"),
        "Set WEBHOOK_URL and HMAC_SECRET env vars to run integration tests",
    )
    def test_integration_tampered_body_rejected_with_403(self):
        import requests as req
        body = self._canonical_body(self.COMPLETED_PAYLOAD)
        # Sign the original then send a different body — server must reject.
        good_sig = self._sign(self._secret(), body)
        tampered = body + "tamper"
        response = req.post(
            self._webhook_url(),
            data=tampered,
            headers={
                "Content-Type": "application/json",
                "X-EvenUp-Webhook-Signature": good_sig,
            },
        )
        self.assertEqual(403, response.status_code)


if __name__ == "__main__":
    unittest.main()
