from generate import chat
import gradio as gr
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history, top_p):
    _, history = chat(history, top_p=top_p)
    return history


def web_demo():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter text and press enter")
        clear = gr.Button("Clear")

        # max_history_len = gr.Slider(0, 5, value=2, step=1, label="Maximum history length", interactive=True)
        # repetition_penalty = gr.Slider(1.0, 3.0, value=1.0, step=0.05, label="Repetition penalty", interactive=True)
        # temperature = gr.Slider(0, 1, value=1, step=0.01, label="Temperature", interactive=True)
        top_p = gr.Slider(0, 1, value=1, step=0.01, label="Top p", interactive=True)
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, top_p], chatbot
        )
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch(share=True)


if __name__ == '__main__':
    web_demo()
