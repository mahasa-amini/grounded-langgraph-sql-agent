import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.config import Settings


class SettingsDebugEnvironmentTests(unittest.TestCase):
    def test_inherited_global_debug_value_is_ignored(self) -> None:
        with patch.dict(os.environ, {"DEBUG": "release"}, clear=True):
            settings = Settings(_env_file=None)

        self.assertTrue(settings.debug)

    def test_project_debug_boolean_values_are_parsed(self) -> None:
        for raw_value, expected in (("true", True), ("false", False)):
            with self.subTest(raw_value=raw_value):
                with patch.dict(
                    os.environ,
                    {
                        "DEBUG": "release",
                        "GROUNDED_SQL_AGENT_DEBUG": raw_value,
                    },
                    clear=True,
                ):
                    settings = Settings(_env_file=None)

                self.assertIs(settings.debug, expected)

    def test_invalid_project_debug_value_raises_validation_error(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEBUG": "release",
                "GROUNDED_SQL_AGENT_DEBUG": "release",
            },
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)


if __name__ == "__main__":
    unittest.main()
