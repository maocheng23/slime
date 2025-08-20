from datasets import load_dataset

ds = load_dataset("JoeYing/ReTool-SFT")["train"]

def convert(sample):
    conversations = sample["conversations"]

    def convert_role(role):
        if role == "human":
            return "user"
        elif role == "gpt":
            return "assistant"
        elif role == "system":
            return "system"
        else:
            raise ValueError(f"Unknown role: {role}")

    messages = [
        {
            "role": convert_role(turn["from"]),
            "content": turn["value"],
        }
        for turn in conversations
    ]

    return {"messages": messages}

ds = ds.map(convert)
ds.to_parquet("./data/retool/ReTool-SFT.parquet")