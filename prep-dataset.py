from datasets import Dataset

import pandas as pd

INSTRUCTION = "Summarize the following story in my style"

df_instructions = pd.DataFrame(columns=['text'])
df_stories = pd.DataFrame(columns=['text'])
df_summaries = pd.DataFrame(columns=['text'])

def combine_texts(instruction, story, summary):
  return {
      "text": f"""
{instruction}

### Story
{story}

### Summary
{summary}
"""}

for pair in FILES:
  with open(pair.format("story"), "r") as file:
    story = "".join(file.readlines())
  with open(pair.format("summary"), "r") as file:
    summary = "".join(file.readlines())

  df_instructions = pd.concat(
    [df_instructions, pd.DataFrame([{'text': INSTRUCTION}])],
    ignore_index=True
  )
  df_stories = pd.concat(
    [df_stories, pd.DataFrame([{'text': story}])],
    ignore_index=True
  )
  df_summaries = pd.concat(
    [df_summaries, pd.DataFrame([{'text': summary}])],
    ignore_index=True
  )

combined_texts = [combine_texts(instruction, story, summary) for instruction, story, summary in zip(df_instructions["text"], df_stories["text"], df_summaries["text"])]

finetuning_dataset = Dataset.from_dict({"text": [ct["text"] for ct in combined_texts]})

finetuning_dataset[0]['text']