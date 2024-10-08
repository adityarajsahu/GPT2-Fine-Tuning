import pandas as pd
from transformers import TextDataset

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    questions = df["Question"].tolist()
    answers = df["Answer"].tolist()
    
    processed_data = []
    for ques, ans in zip(questions, answers):
        ques = ques.replace("[INST]", "")
        combined_qna = "Q: {}\nA: {}".format(ques, ans)
        processed_data.append(combined_qna)
    return processed_data

def save_text_file(data, file_path):
    with open(file_path, "w") as file:
        for line in data:
            file.write(line + "\n")

def prepare_dataset(tokenizer, dataset_file, block_size = 128):
    return TextDataset(
        tokenizer = tokenizer,
        file_path = dataset_file,
        block_size = block_size
    )