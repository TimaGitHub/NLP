from typing import Optional
import torch
import time
import json
from sentencepiece import SentencePieceProcessor
from pathlib import Path

from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args


    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0, "No checkpoint found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f}s")
            prev_time = time.time()
        with open(Path(checkpoint_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in{(time.time() - prev_time):.2f}")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: str, temperature: float = -0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)

        assert batch_size <= self.args.max_batch_size

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        #for k, t in enumerate(prompt_tokens):



if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False

    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LLaMA.build(
        checkpoint_dir="Llama-2-7b/",
        tokenizer_path="Llama-2-7b/tokenizer.model",
        load_model=True,
        max_batch_size=1,
        max_seq_len=1024,
        device=device
    )

    print('All ok')

