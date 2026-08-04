# emg-kd


# Teacher: Gaddy 2021 recognition model

Checkpoint: Zenodo 10.5281/zenodo.7183877

- params: 54.12M
- input: raw EMG only, (batch, time, 8), 1000 Hz
- call signature: model(None, x_raw, None)  -- x_feat and session_ids unused
- conv blocks: 3 x ResBlock, stride 2 each, 768 ch
- downsample: 8x (verified: 4000 -> 500, 8000 -> 1000)
- transformer: 6 layers, d_model 768, 8 heads, head dim 96, FFN 3072
- output: 38 logits (37 chars + CTC blank at index 37)
- session embedding: NONE (absent from weights and forward)

## Traps
- model.train() enables random time shift of up to 8 samples.
  Teacher must stay in eval() for logit caching.
- Gaddy's MultiHeadAttention lacks .batch_first; patch it after load.