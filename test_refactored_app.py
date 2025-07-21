import unittest
from unittest.mock import MagicMock, patch
from streamlit_app import main


class TestRefactoredApp(unittest.TestCase):
    @patch("streamlit_app.st")
    @patch("streamlit_app.OpenAI")
    def test_title_and_description(self, mock_openai, mock_st):
        """Test that the title and description are correct."""
        main()
        mock_st.title.assert_called_with("💬 Chatbot")
        mock_st.write.assert_called_with(
            "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
            "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
            "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
        )

    @patch("streamlit_app.st")
    def test_openai_api_key_input_not_provided(self, mock_st):
        """Test that the app shows an info message if the API key is not provided."""
        mock_st.text_input.return_value = ""
        main()
        mock_st.info.assert_called_with(
            "Please add your OpenAI API key to continue.", icon="🗝️"
        )

    @patch("streamlit_app.st")
    @patch("streamlit_app.OpenAI")
    def test_chat_functionality(self, mock_openai, mock_st):
        """Test that the chat functionality works as expected."""
        mock_st.text_input.return_value = "fake_api_key"
        mock_st.chat_input.return_value = "Hello"
        mock_st.session_state.messages = []
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_stream = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_st.write_stream.return_value = "Hi there!"

        main()

        self.assertEqual(
            mock_st.session_state.messages,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
