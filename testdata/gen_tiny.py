"""Builds the tiny LFM2 checkpoint the lfm tests run on: LFM2.5-350M's architecture and its real
65 536-token vocabulary, tiny sizes, random weights. The embedding row of <|im_end|> (id 7) is
scaled so a greedy answer ends after a few tokens, which lets the tests see the end of a turn.

    ~/Dev/LMmodels/.venv/bin/python testdata/gen_tiny.py ~/Dev/LMmodels/LiquidAI/LFM2.5-350M SEED SCALE
    weightsc -in testdata/tiny_build -out testdata/tiny_lfm2.wtypw -merges-out testdata/tiny_lfm2.merges \
        -id tiny-lfm2 -version 1 -quant int8-block32 -prefix model.

testdata/tiny_build/ is a scratch directory (not committed).
"""
import json, os, shutil, sys
import torch
from safetensors.torch import save_file
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

src, seed, scale = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
torch.manual_seed(seed)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_build")
os.makedirs(out, exist_ok=True)

cfg = Lfm2Config(
    vocab_size=65536, hidden_size=32, intermediate_size=64, block_auto_adjust_ff_dim=False,
    num_hidden_layers=4, layer_types=["conv", "full_attention", "conv", "full_attention"],
    num_attention_heads=4, num_key_value_heads=2, conv_L_cache=3, conv_bias=False,
    norm_eps=1e-5, tie_word_embeddings=True, max_position_embeddings=4096,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}, dtype="float32",
)
model = Lfm2ForCausalLM(cfg).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1.0 + torch.empty_like(p).uniform_(-0.2, 0.2))
        else:
            p.copy_(torch.randn_like(p) * 0.08)
    model.model.embed_tokens.weight[7] *= scale

state = {k: v.contiguous().clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
save_file(state, os.path.join(out, "model.safetensors"))
json.dump({k: v for k, v in cfg.to_dict().items() if not k.startswith("_")},
          open(os.path.join(out, "config.json"), "w"), indent=1, default=str)
for f in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(os.path.join(src, f), out)
print("ok", seed, scale)
