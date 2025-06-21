from idlelib.run import flush_stdout

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from typing import Optional, Tuple, Dict, Any
from safetensors.torch import load_file

from src.tokenizer import tokenizer


class MyGPTConfig(PretrainedConfig):
    model_type = "my_gpt"

    def __init__(self,
                 vocab_size=50257,  # Default to GPT-2 vocab size, replace with your actual
                 pad_token_id: Optional[int] = None,  # Hugging Face uses pad_token_id
                 n_layer=20,
                 n_head=20,
                 d_model=1280,
                 dropout=0.06,
                 max_position_embeddings=1024,  # Renamed from max_len for HF convention
                 bos_token_id=50256,  # Example, replace if needed
                 eos_token_id=50256,  # Example, replace if needed
                 **kwargs):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


# --- Your GPT Model adapted for Hugging Face ---
class MyGPTModel(PreTrainedModel):
    config_class = MyGPTConfig  # Link to your config

    def __init__(self, config: MyGPTConfig):
        super().__init__(config)
        self.config = config  # Store config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.d_model))

        self.blocks = nn.ModuleList([
            self._build_block() for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (standard practice)
        self.lm_head.weight = self.tok_emb.weight

        # Causal mask is usually handled by Attention class or generate method,
        # but we can define it if your nn.MultiheadAttention needs it explicitly.
        # For HF generate, it's better to let it handle masks or pass attention_mask.
        # self.register_buffer(
        # "causal_mask",
        # torch.triu(torch.ones(config.max_position_embeddings, config.max_position_embeddings, dtype=torch.bool), 1)
        # )
        # Note: nn.MultiheadAttention takes attn_mask where True indicates masking.
        # So, for causal, you'd want True for future positions.

        # Initialize weights
        self.post_init()  # HF specific method for final initializations

    def _build_block(self):
        d, h, drop = self.config.d_model, self.config.n_head, self.config.dropout
        return nn.ModuleDict({
            "ln1": nn.LayerNorm(d),
            "attn": nn.MultiheadAttention(d, h, dropout=drop, batch_first=True),
            "ln2": nn.LayerNorm(d),
            "mlp": nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, d),
                nn.Dropout(drop)
            )
        })

    def _init_weights(self, module):
        """ Initializes weights of the model. """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # For positional embeddings, often initialized differently if learned
        if isinstance(module, MyGPTModel):  # Initialize pos_emb for the model itself
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, new_embeddings):
        self.tok_emb = new_embeddings
        self.lm_head.weight = new_embeddings.weight  # Re-tie weights

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs) -> Dict[
        str, Any]:
        # This function is crucial for .generate()
        # It prepares model inputs for the next generation step.

        # if past is defined in model_kwargs, use it
        # `past_key_values` is a tuple of tuples, one for each layer.
        # Each inner tuple contains (key_states, value_states) for self-attention
        # key_states, value_states shape: (batch_size, num_heads, sequence_length, embed_size_per_head)
        # For nn.MultiheadAttention with batch_first=True, cache is usually (batch_size, seq_len, embed_dim)
        # Let's assume for now our custom model doesn't have explicit KV caching in this format for simplicity,
        # and relies on recomputing. If you implement KV caching, this function needs to handle it.

        # If past_key_values are used, we only need the last token of input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,  # Pass it along, even if not explicitly used by this simplified forward
            "attention_mask": attention_mask,
            # "use_cache": kwargs.get("use_cache", False) # Important for efficient generation
        }

    def forward(
            self,
            input_ids: torch.LongTensor,  # [B, T]
            attention_mask: Optional[torch.Tensor] = None,  # [B, T] (0 for padding, 1 for token)
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,  # For KV caching
            labels: Optional[torch.LongTensor] = None,  # For training
            use_cache: Optional[bool] = None,  # Whether to use KV caching
            output_attentions: Optional[bool] = None,  # Not implemented here
            output_hidden_states: Optional[bool] = None,  # Not implemented here
            return_dict: Optional[bool] = None,  # Whether to return a dict or tuple
    ) -> Tuple[torch.Tensor, ...]:

        B, T = input_ids.shape
        # For Hugging Face, `max_position_embeddings` is the name in config
        assert T <= self.config.max_position_embeddings, \
            f"Sequence length {T} exceeds model maximum length {self.config.max_position_embeddings}"

        # Token embeddings + positional embeddings
        tok_embs = self.tok_emb(input_ids)  # [B, T, d_model]
        pos_embs = self.pos_emb[:, :T, :]  # [1, T, d_model]
        x = tok_embs + pos_embs  # [B, T, d_model]

        # Create causal mask if not provided by attention_mask handling in generate
        # Standard Hugging Face attention_mask: 1 for tokens to attend to, 0 for padding.
        # For causal, nn.MultiheadAttention needs a mask where True means "skip".
        # It should be of shape (T, T) or (B * n_head, T, T)
        # The `generate` method typically creates the appropriate combined causal and padding mask.
        # If you pass `attention_mask` from `input_ids`, it's typically for padding.
        # Let's construct a causal mask compatible with `nn.MultiheadAttention`
        # This is a square mask for the sequence length `T`
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)

        # If a padding `attention_mask` is provided by `generate`, we might need to combine it.
        # `key_padding_mask` for `nn.MultiheadAttention` should be (B, T) where True means "mask this position".
        # So, if `attention_mask` is (1=attend, 0=pad), then `key_padding_mask = (attention_mask == 0)`.
        key_padding_mask = None
        if attention_mask is not None:
            # Assuming attention_mask from HF is 1 for non-padded, 0 for padded
            key_padding_mask = (attention_mask == 0)
            if key_padding_mask.all():  # If all are padded, something is wrong or it's an empty sequence.
                key_padding_mask = None  # Avoid issues with all-True masks in some attention impl.

        # Transformer blocks
        # KV Caching is not explicitly implemented here to keep it simple.
        # For efficient generation, you'd store and reuse key/value projections from previous steps.
        for blk in self.blocks:
            attn_output, _ = blk["attn"](
                query=blk["ln1"](x),
                key=blk["ln1"](x),  # Self-attention
                value=blk["ln1"](x),  # Self-attention
                attn_mask=causal_mask,  # For causal attention
                key_padding_mask=key_padding_mask,  # For padding
                need_weights=False  # We don't need attention weights for generation logits
            )
            x = x + attn_output
            x = x + blk["mlp"](blk["ln2"](x))

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if labels is not None:
            # Calculate loss if labels are provided
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            # if past_key_values are handled, they should be returned here for use_cache=True
            return ((output + (None,)) if loss is None else (loss,) + output + (None,))

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Not returning past_key_values in this simplified version
            hidden_states=None,  # Not returning hidden_states
            attentions=None,  # Not returning attentions
        )


def generate_text(
        prompt: str,
        merged_ckpt: str,  # path to gpt_sft_merged_stepXXXX.safetensors
        cfg_kwargs: Dict[str, Any],  # same hyper-params you used during training
        *,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
) -> str:
    """
    Inference helper for the *merged* model (no LoRA / PEFT needed).
    Works on CPU/MPS – no CUDA or bits-and-bytes required.
    """

    # ── build empty model skeleton ───────────────────────────────────────────
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"

    cfg_kwargs = cfg_kwargs.copy()  # don't mutate caller's dict
    cfg_kwargs.update(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = MyGPTModel(MyGPTConfig(**cfg_kwargs))
    DEVICE = 'cpu'
    #     "mps" if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    print(f"🔄 loading merged weights from {merged_ckpt} → {DEVICE}")
    model.load_state_dict(load_file(merged_ckpt, device='cpu'), strict=False)
    model.to(DEVICE).eval()

    # ── tokenise prompt ──────────────────────────────────────────────────────
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # ── generate ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )[0]

    print("\n--- Generated text -------------------------------------------")
    print(tokenizer.decode(out, skip_special_tokens=True))
    print("--------------------------------------------------------------")
    return tokenizer.decode(out, skip_special_tokens=True)


def generate_text_old(
        model_weights_path: str,
        prompt: str,
        cfg_kwargs: dict,
        max_new_tokens=120,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        repetition_penalty=1.1
):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    # Ensure pad_token_id is set for the tokenizer if it doesn't have one
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            print(
                f"Tokenizer does not have a pad_token_id. Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Add a pad token if absolutely necessary, though this might affect model performance if not trained with it
            print("Tokenizer has no pad_token_id and no eos_token_id. Adding a new pad token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # The config_params vocab_size might need to be updated if a token is added
            # config_params['vocab_size'] = len(tokenizer) # This is risky if model wasn't trained with it
    # Update config_params with tokenizer specifics if necessary
    cfg_kwargs['vocab_size'] = tokenizer.vocab_size
    cfg_kwargs['pad_token_id'] = tokenizer.pad_token_id
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        cfg_kwargs['bos_token_id'] = tokenizer.bos_token_id
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        cfg_kwargs['eos_token_id'] = tokenizer.eos_token_id

    # 2. Initialize Configuration
    print(f"Initializing model with config: {cfg_kwargs}")
    config = MyGPTConfig(**cfg_kwargs)

    # 3. Initialize Model
    print("Initializing model structure...")
    model = MyGPTModel(config)

    # 4. Load Model Weights
    try:
        print(f"Loading model weights from: {model_weights_path}")
        # If your checkpoint was saved with model.module.state_dict() (from DataParallel or DDP)
        # you might need to adjust the keys.
        state_dict = torch.load(model_weights_path, map_location='cpu')  # Load to CPU first

        # Handle potential 'module.' prefix if saved from nn.DataParallel or DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove `module.`
            else:
                new_state_dict[k] = v

        # Special handling for pos_emb if its size in checkpoint differs from config
        # This can happen if max_len was different during training vs. inference config
        if 'pos_emb' in new_state_dict and new_state_dict['pos_emb'].shape != model.pos_emb.shape:
            print(
                f"Warning: Positional embedding size mismatch. Checkpoint: {new_state_dict['pos_emb'].shape}, Model: {model.pos_emb.shape}")
            print("Attempting to adjust positional embeddings. This is a common step if max_len changed.")
            # Take the slice of the checkpoint's pos_emb that fits the model's pos_emb
            cp_pos_emb = new_state_dict['pos_emb']
            model_pos_emb_len = model.pos_emb.shape[1]
            cp_pos_emb_len = cp_pos_emb.shape[1]

            # Truncate or pad the checkpoint's positional embeddings
            if model_pos_emb_len <= cp_pos_emb_len:
                new_state_dict['pos_emb'] = cp_pos_emb[:, :model_pos_emb_len, :]
            else:  # model_pos_emb_len > cp_pos_emb_len (model needs more, pad with zeros or reinit)
                # This case is trickier and might require more sophisticated handling or re-training.
                # For simplicity, we'll pad with zeros from a newly initialized pos_emb.
                print(
                    f"Model's max_position_embeddings ({model_pos_emb_len}) is larger than checkpoint's ({cp_pos_emb_len}). Padding with newly initialized values.")
                temp_pos_emb = torch.zeros_like(model.pos_emb)
                torch.nn.init.normal_(temp_pos_emb, mean=0.0, std=0.02)  # Re-initialize like in _init_weights
                temp_pos_emb[:, :cp_pos_emb_len, :] = cp_pos_emb[:, :cp_pos_emb_len, :]
                new_state_dict['pos_emb'] = temp_pos_emb

        # Load the state dict
        model.load_state_dict(new_state_dict, strict=False)  # Use strict=False initially for debugging
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model_weights_path is correct and the state_dict keys match the model architecture.")
        return

    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                       max_length=config.max_position_embeddings)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    print(f"\nGenerating text with parameters:")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  do_sample: {do_sample}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  temperature: {temperature}")
    print(f"  repetition_penalty: {repetition_penalty}\n")
    print(f"Prompt: \"{prompt}\"")

    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    print("\n--- Generated Text ---")
    print(full_text)
    print("--- End of Generation ---")


if __name__ == "__main__":
    ckpt_path = "model.safetensors"
    cfg = dict(n_layer=20, n_head=20, d_model=1280, vocab_size=50257,
               dropout=0.06, max_position_embeddings=512)

    SYSTEM = "<|system|>\nYou are a helpful assistant.\n"
    U_TAG = "<|user|>\n"
    A_TAG = "<|assistant|>\n"

    prompt = (
        "For the following story, how much money did Christopher find "
        "in his pocket?\n\n"
        "<User> Christopher went outside on his porch this morning. "
        "When it started to rain, he put on his raincoat. "
        "He reached in the pocket and found ten dollars. "
        "He was very surprised!"
    )

    full_prompt = f"{SYSTEM}{U_TAG}{prompt}\n{A_TAG}"

    print(generate_text_old(
        prompt=full_prompt,
        model_weights_path='checkpoint150.pth',
        # merged_ckpt=ckpt_path,
        cfg_kwargs=cfg,
        max_new_tokens=80,
        temperature=0.7,
    ))
