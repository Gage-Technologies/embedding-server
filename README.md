<div align="center">

# Embedding Server

<a href="https://github.com/gage-technologies/embedding-server">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/gage-technologies/embedding-server?style=social">
</a>
<a href="https://github.com/gage-technologies/embedding-server/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/github/license/gage-technologies/embedding-server">
</a>
<a href="https://huggingface.github.io/text-generation-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

</div>

A Rust, Python and gRPC server for sentence-transformer embeddings.

## Overview
This is the beginning of the repo so we're pretty light on the info. More will come! The basic gist is that we intend to 
create the equivalent of Huggingface's Text Generation Inference API but for sentence-transformer embeddings. This repo 
is a fork of the [text-generation-inference](https://github.com/huggingface/text-generation-inference) repo.

The current state of the repo is that we have forked the codebase to run the base Sentence-Transformer class from the 
[sentence-transformers](https://github.com/UKPLab/sentence-transformers) library. The server will run manually but we 
have not completed the dockerfile or actions to create a build pipeline. More to come!

## Develop

```shell
make server-dev
make router-dev
```

## Testing

```shell
# python
make python-server-tests
make python-client-tests
# or both server and client tests
make python-tests
# rust cargo tests
make rust-tests
# integration tests
make integration-tests
```
