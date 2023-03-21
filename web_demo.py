from interact import chat, set_args
import gradio as gr
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
from utils import fix_seed

args = set_args()
fix_seed(42)
kwargs = {"device": args.device, "max_len": args.max_len, "topk": args.topk, "topp": args.topp}
tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
model = GPT2LMHeadModel.from_pretrained(args.model_path).to(args.device)
model.eval()


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history, max_history_len, repetition_penalty, temperature):
    history = chat(model, tokenizer, history, max_history_len, repetition_penalty, temperature, **kwargs)
    return history


def web_demo():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter text and press enter")
        clear = gr.Button("Clear")

        max_history_len = gr.Slider(0, 5, value=2, step=1, label="Maximum history length", interactive=True)
        repetition_penalty = gr.Slider(1.0, 3.0, value=1.0, step=0.05, label="Repetition penalty", interactive=True)
        temperature = gr.Slider(0, 1, value=1, step=0.01, label="Temperature", interactive=True)
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, max_history_len, repetition_penalty, temperature], chatbot
        )
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch(share=True)


if __name__ == '__main__':
    web_demo()
