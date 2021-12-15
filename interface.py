import gradio as gr
import pickle


def test(text):
    return "This statement is false/true."


input1 = gr.inputs.Textbox(label="Sentence to check")

gui = gr.Interface(fn=test,
                   inputs=input1,
                   outputs="text")

gui.launch()
