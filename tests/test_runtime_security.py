import unittest

from memer.security import RuntimeSecurity


class RuntimeSecurityTests(unittest.TestCase):
    def test_authentication_is_optional_for_local_development(self):
        guard = RuntimeSecurity()
        self.assertTrue(guard.is_authorized(None))

    def test_accepts_configured_bearer_key(self):
        guard = RuntimeSecurity(("a" * 40,), require_api_key=True)
        self.assertTrue(guard.is_authorized("Bearer " + "a" * 40))

    def test_rejects_missing_wrong_or_malformed_key(self):
        guard = RuntimeSecurity(("a" * 40,), require_api_key=True)
        for header in (None, "", "Basic " + "a" * 40, "Bearer wrong"):
            with self.subTest(header=header):
                self.assertFalse(guard.is_authorized(header))

    def test_generation_slots_are_bounded(self):
        guard = RuntimeSecurity(max_concurrent_generations=1)
        self.assertTrue(guard.acquire_generation_slot())
        self.assertFalse(guard.acquire_generation_slot())
        guard.release_generation_slot()
        self.assertTrue(guard.acquire_generation_slot())
        guard.release_generation_slot()


if __name__ == "__main__":
    unittest.main()
