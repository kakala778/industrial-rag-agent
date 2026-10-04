import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitExampleTaskTests(unittest.TestCase):
    APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

    def test_example_task_buttons_fill_task_without_running_agent(self):
        examples = (
            (
                "Operating Environment Requirements",
                "Find the operating-environment requirements in the selected specification.",
            ),
            (
                "Equipment Configuration",
                "Find a target equipment row's quantity and configuration in the structured schedule.",
            ),
            (
                "Check Unsupported Information",
                "Check whether [specify one protocol or security parameter] is supported by the selected specification.",
            ),
        )

        for label, expected_task in examples:
            with self.subTest(example=label):
                app = AppTest.from_file(self.APP_PATH).run()
                matching_buttons = [button for button in app.button if button.label == label]
                self.assertEqual(len(matching_buttons), 1)

                matching_buttons[0].click().run()

                self.assertEqual(app.text_area(key="research_task").value, expected_task)
                self.assertNotIn("latest_research_run", app.session_state)
                self.assertEqual(len(app.exception), 0)

    def test_selecting_example_clears_stale_research_result(self):
        app = AppTest.from_file(self.APP_PATH).run()
        app.session_state["latest_research_run"] = {"error": "previous result"}
        button = next(
            item for item in app.button
            if item.label == "Equipment Configuration"
        )

        button.click().run()

        self.assertNotIn("latest_research_run", app.session_state)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
