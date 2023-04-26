from generate import chat
import gradio as gr


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history, top_p, temperature):
    _, history = chat(history, top_p=top_p, temperature=temperature)
    return history


def web_demo():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter text and press enter")
        clear = gr.Button("Clear")

        # max_history_len = gr.Slider(0, 5, value=2, step=1, label="Maximum history length", interactive=True)
        # repetition_penalty = gr.Slider(1.0, 3.0, value=1.0, step=0.05, label="Repetition penalty", interactive=True)
        temperature = gr.Slider(0, 2, value=0.7, step=0.05, label="Temperature", interactive=True)
        top_p = gr.Slider(0, 1, value=0.9, step=0.01, label="Top p", interactive=True)
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, top_p, temperature], chatbot
        )
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch(share=True)


if __name__ == '__main__':
    web_demo()
