# AutoTimes

The repo includes the detailed implementation for the paper: AutoTimes: Autoregressive Time Series Forecasters via Large Language Models. 

You can download the datasets from other repositories like Autoformer.

Large language models can be downloaded from Hugging Face.

   * [GPT2](https://huggingface.co/openai-community/gpt2)
   * [OPT Family](https://huggingface.co/facebook/opt-125m)
   * [LLaMA-7B](https://huggingface.co/meta-llama/Llama-2-7b)

## Usage 

```
# the default large language model is LLaMA-7B

# long-term forecasting
bash ./scripts/time_series_forecasting/long_term/AutoTimes_ETTh1.sh

# short-term forecasting
bash ./scripts/time_series_forecasting/short_term/AutoTimes_M4.sh

# zero-shot forecasting
# it's worth noting that sM4_tM3 utilizes models trained
# on short-term, you should run AutoTimes_M4 first
bash ./scripts/zero_shot_forecasting/sM4_tM3.sh
bash ./scripts/zero_shot_forecasting/sM3_tM4.sh

# in-context forecasting
bash ./scripts/in_context_forecasting/M3.sh

# other large language models
bash ./scripts/method_generality/opt.sh

# preprocess timestamps to generate text embedding
python ./preprocess.py --gpu 0 --dataset ETTh1
```

