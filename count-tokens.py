from transformers import AutoTokenizer

FILES = [
    "data/{}-1.txt",
    "data/{}-2.txt",
    "data/{}-3.txt",
    "data/{}-4.txt",
    "data/{}-5.txt",
]
INSTRUCTION = "Summarize the following story in my style"

tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit"
)

print(f"{'Total':<12}{'Instruction':<12}{'Story':<12}{'Summary':<12}")
for pair in FILES:
  with open(pair.format("story"), "r") as file:
    story = "".join(file.readlines())
  with open(pair.format("summary"), "r") as file:
    summary = "".join(file.readlines())

  # Count tokens
  instruction_tokens = tokenizer(INSTRUCTION, return_tensors="pt")["input_ids"].shape[1]
  story_tokens = tokenizer(story, return_tensors="pt")["input_ids"].shape[1]
  summary_tokens = tokenizer(summary, return_tensors="pt")["input_ids"].shape[1]
  
  # Print table of tokens
  total_tokens = instruction_tokens + story_tokens + summary_tokens
  print(f"{total_tokens:<12}{instruction_tokens:<12}{story_tokens:<12}{summary_tokens:<12}")
