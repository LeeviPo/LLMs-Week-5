# LLMs-Week-5
Coursework

The task was to find errors in a given Jupyter Notebook, fix them, and then run the book which fine-tunes a small model (Phi-2 in this case). The fine-tuned model was uploaded to Hugging Face.


## Errors
**Trying to get rouge_score version using the version attribute, which it doesn't have**
```
# Original
import rouge_score

print("Rouge Score version:", rouge_score.__version__)


# Fixed
import rouge_score
from importlib_metadata import version # Using importlib_metadata to get version of rouge_score

print("Rouge Score version:", version("rouge_score"))
```

**Loading checkpoint that 1000, but the model is trained only for 500 steps as per instructions**
```
# Original
ft_model = PeftModel.from_pretrained(
    base_model, 
    "/kaggle/working/peft-dialogue-summary-training/final-checkpoint/checkpoint-1000",
    torch_dtype=torch.float16,
    is_trainable=False
)


# Fixed
ft_model = PeftModel.from_pretrained(
    base_model, 
    "/kaggle/working/peft-dialogue-summary-training/final-checkpoint/checkpoint-500",
    torch_dtype=torch.float16,
    is_trainable=False
)
```


