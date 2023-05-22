<div align="center">

# Text Generation Inference benchmarking tool

![benchmark](../assets/benchmark.png)

</div>

A lightweight benchmarking tool based inspired by [oha](https://github.com/hatoo/oha) 
and powered by [tui](https://github.com/tui-rs-revival/ratatui).

## Install 

```shell 
make install-benchmark
```

## Run

First, start `text-generation-inference`:

```shell
text-generation-launcher --model-id all-MiniLM-L6-v2
```

Then run the benchmarking tool:

```shell
text-generation-benchmark --tokenizer-name all-MiniLM-L6-v2
```