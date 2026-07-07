# LLM4TSF

Large Language Models · Time Series Forecasting · Pre-alignment · Post-alignment · Cross-dataset Learning

## 📰 Paper

### Introduction

Time series forecasting (TSF) is a fundamental task across many real-world domains. Recently, Large Language Models (LLMs) have been introduced into TSF to leverage their pretrained knowledge and representation capacity. However, whether LLMs are truly useful for forecasting remains debated.

This project studies the role of LLMs in time series forecasting through a large-scale empirical analysis. We compare two mainstream LLM4TSF alignment strategies, **pre-alignment** and **post-alignment**, under both single-dataset and cross-dataset learning settings.

Our study shows that LLM4TSF can improve forecasting performance when trained with diverse time series data and appropriate alignment strategies. In particular, pre-alignment often performs better than post-alignment, while pretrained LLM parameters are especially useful under distribution shift and out-of-domain scenarios.

### Alignment Strategies

#### Pre-alignment

Pre-alignment maps time-series embeddings into the LLM word embedding space before feeding them into the LLM. The LLM is usually frozen, and the model mainly trains the time-series encoder, alignment module, and forecasting decoder.

#### Post-alignment

Post-alignment jointly feeds time-series embeddings and textual prompt embeddings into the LLM. The alignment between time series and text is learned inside the LLM representation space through supervised forecasting.

### Experimental Setup

We evaluate LLM4TSF across large-scale and diverse forecasting scenarios, including:

* 62 real-world time series datasets
* Over 8 billion observations
* More than 10 application domains
* 17 forecasting scenarios
* In-domain and out-of-domain evaluation
* Forecasting horizons of 96, 192, 336, and 720

The models use GPT-2 as the LLM backbone and are evaluated using MAE and MSE.

### Main Findings

Our experiments lead to several key observations:

* LLM4TSF benefits from diverse cross-dataset training.
* Pre-alignment outperforms post-alignment in most tasks.
* Pretrained LLM parameters improve performance, especially under distribution shifts.
* Removing or randomly initializing the LLM often degrades performance.
* LLMs are more useful for time series with strong shifting and frequent transitions.
* Larger LLMs do not automatically lead to better forecasting performance.
* Informative textual prompts are important for improving LLM4TSF performance.

### Code Structure

```text
LLM4TSF/
├── src/
│   └── models/
│       ├── pre_alignment.py
│       ├── post_alignment.py
│       ├── patch_embedding.py
│       ├── ts_encoder.py
│       ├── ts_decoder.py
│       └── alignment/
│           └── pca_aligner.py
├── test_pre_alignment.py
├── test_post_alignment.py
├── requirements.txt
└── README.md
```


### Citation

```bibtex
@article{qiu2026rethinking,
  title={Rethinking the role of llms in time series forecasting},
  author={Qiu, Xin and Tong, Junlong and Sun, Yirong and Ma, Yunpu and Zhang, Wei and Shen, Xiaoyu},
  journal={arXiv preprint arXiv:2602.14744},
  year={2026}
}
```

### Contact

For questions or collaborations, please contact us at [qiuxinzju@zju.edu.cn](mailto:qiuxinzju@zju.edu.cn).
