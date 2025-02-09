import gradio as gr

def add_dict(dict_list):
    dict_list.append({"key": "", "value": ""})
    return dict_list

with gr.Blocks(css="""
.dataframe-wrap, .dataframe-scroll {
    overflow: hidden !important;
    height: auto !important;
    max-height: none !important;
}
""") as demo:
    state = gr.State(value=[{"key": "k1", "value": "v1"}])
    dict_list = gr.DataFrame(
        value=[["k1", "v1"]],
        headers=["key", "value"],
        row_count=(1, "dynamic"),
        col_count=2,
        interactive=True,
        label="字典列表"
    )
    add_button = gr.Button("＋ 添加新字典")
    add_button.click(
        add_dict,
        inputs=state,
        outputs=state
    ).then(
        lambda d: [[item["key"], item["value"]] for item in d],
        inputs=state,
        outputs=dict_list
    )
demo.launch()