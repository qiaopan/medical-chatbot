import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest


class GroqConfigurationTests(unittest.TestCase):
    def test_missing_key_has_clear_error(self):
        from optimal import connect_memory_with_llm_category_env as app

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Missing GROQ_API_KEY"):
                app.get_llm()

    def test_llm_uses_configured_groq_model(self):
        from optimal import connect_memory_with_llm_category_env as app

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True):
            llm = app.get_llm()
        self.assertEqual(llm.model_name, app.GROQ_MODEL)

    def test_classifier_import_does_not_require_key(self):
        with patch.dict(os.environ, {}, clear=True):
            module = importlib.import_module("optimal.category.classify_text_to_json")
        self.assertTrue(callable(module.anonymize_text))


class RagPipelineTests(unittest.TestCase):
    def test_retrieval_loads_index_and_returns_sources(self):
        from optimal import connect_memory_with_llm_category_env as app

        docs = app.retrieve_documents(
            app.load_vectorstore(),
            "What vaccinations are recommended during pregnancy?",
        )
        self.assertEqual(len(docs), 6)
        self.assertTrue(all(doc.page_content for doc in docs))
        self.assertTrue(all(doc.metadata.get("source") for doc in docs))

    def test_answer_prompt_preserves_citations(self):
        from optimal import medibot_category_env as app

        class FakeLlm:
            def invoke(self, prompt):
                self.prompt = prompt
                return AIMessage(content="Grounded answer [1]")

        docs = app.retrieve_documents(
            app.get_vectorstore(),
            "What vaccinations are recommended during pregnancy?",
        )
        llm = FakeLlm()
        answer = app.generate_answer(llm, "pregnancy vaccines", docs, {})
        sources = app.format_source_documents(docs)
        self.assertEqual(answer, "Grounded answer [1]")
        self.assertEqual(len(sources), 6)
        self.assertIn("Citation:", llm.prompt)
        self.assertTrue(sources[0]["title"].startswith("[1]"))


class FrontendTests(unittest.TestCase):
    def test_streamlit_page_renders(self):
        app_path = Path(__file__).parents[1] / "optimal" / "medibot_category_env.py"
        app = AppTest.from_file(str(app_path)).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Ask Chatbot!")
        self.assertEqual(len(app.chat_input), 1)


class LiveGroqTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_LIVE_GROQ_TESTS") == "1",
        "Set RUN_LIVE_GROQ_TESTS=1 to spend one small live Groq request.",
    )
    def test_api_key_and_model_are_valid(self):
        from optimal import connect_memory_with_llm_category_env as app

        response = app.get_llm(temperature=0, max_tokens=128).invoke(
            "Reply with exactly: OK"
        )
        self.assertIn("OK", response.content.upper())


if __name__ == "__main__":
    unittest.main()
