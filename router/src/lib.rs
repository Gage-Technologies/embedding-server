mod health;
/// Text Generation Inference Webserver
mod infer;
mod queue;
pub mod server;
mod validation;

use infer::Infer;
use queue::{Entry, Queue};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validation::Validation;

/// Hub type
#[derive(Clone, Debug, Deserialize)]
pub struct HubModelInfo {
    #[serde(rename(deserialize = "id"))]
    pub model_id: String,
    pub sha: Option<String>,
    pub pipeline_tag: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "all-MiniLM-L6-v2")]
    pub model_id: String,
    #[schema(nullable = true, example = "e985a63cdc139290c5f700ff1929f0b5942cced2")]
    pub model_sha: Option<String>,
    #[schema(example = "torch.float16")]
    pub model_dtype: String,
    #[schema(example = "cuda")]
    pub model_device_type: String,
    #[schema(nullable = true, example = "text-generation")]
    pub model_pipeline_tag: Option<String>,

    /// Router Parameters
    #[schema(example = "128")]
    pub max_concurrent_requests: usize,
    #[schema(example = "1024")]
    pub max_input_length: usize,
    #[schema(example = "1.2")]
    pub waiting_served_ratio: f32,
    #[schema(example = "32000")]
    pub max_batch_total_tokens: u32,
    #[schema(example = "2")]
    pub validation_workers: usize,

    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct EmbedRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct CompatEmbedRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub stream: bool,
}

impl From<CompatEmbedRequest> for EmbedRequest {
    fn from(req: CompatEmbedRequest) -> Self {
        Self {
            inputs: req.inputs,
        }
    }
}

#[derive(Serialize, ToSchema)]
pub(crate) struct EmbedResponse {
    #[schema(example = "test")]
    pub embedding: Vec<f32>,
    #[schema(example = 768)]
    pub dim: u32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: String,
    pub error_type: String,
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use tokenizers::Tokenizer;

    pub(crate) async fn get_tokenizer() -> Tokenizer {
        if !std::path::Path::new("tokenizer.json").exists() {
            let content = reqwest::get("https://huggingface.co/gpt2/raw/main/tokenizer.json")
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let mut file = std::fs::File::create("tokenizer.json").unwrap();
            file.write_all(&content).unwrap();
        }
        Tokenizer::from_file("tokenizer.json").unwrap()
    }
}
