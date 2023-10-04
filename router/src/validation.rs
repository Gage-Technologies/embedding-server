use flume::SendError;
use crate::validation::ValidationError::{EmptyInput};
/// Payload validation logic
use crate::{EmbedRequest};
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::Encoding;
use tokenizers::TruncationDirection;
use tokio::sync::oneshot;
use tracing::{instrument, Span};

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Validation parameters
    max_input_length: usize,
    /// Channel to communicate with the background tokenization task
    sender: Option<flume::Sender<TokenizerRequest>>,
}

impl Validation {
    pub(crate) fn new(
        workers: usize,
        tokenizer: Option<Tokenizer>,
        max_input_length: usize,
    ) -> Self {
        // If we have a fast tokenizer
        let sender = if let Some(tokenizer) = tokenizer {
            // Create channel
            let (validation_sender, validation_receiver) = flume::unbounded();

            // Create workers
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let receiver_clone = validation_receiver.clone();

                // Spawn worker
                tokio::task::spawn_blocking(move || {
                    tokenizer_worker(tokenizer_clone, receiver_clone)
                });
            }
            Some(validation_sender)
        } else {
            None
        };

        Self {
            sender,
            max_input_length,
        }
    }

    #[instrument(skip_all)]
    async fn validate_input(
        &self,
        inputs: String,
        truncate: Option<usize>,
    ) -> Result<(String, usize), ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender
                .send(((inputs, truncate), response_sender, Span::current()))
                .unwrap();

            // Await on response channel
            // Unwrap is safe here
            let (inputs, _, input_length) = response_receiver.await.unwrap()?;

            // Validate InputLength
            if input_length > self.max_input_length {
                return Err(ValidationError::InputLength(
                    self.max_input_length,
                    input_length,
                ));
            }

            metrics::histogram!("tgi_request_input_length", input_length as f64);
            Ok((inputs, input_length))
        }
        // Return inputs without validation
        else {
            // In this case, we don't know the real length in tokens of the inputs
            // However, the inputs will be truncated by the python servers
            // We make sure that truncate + max_new_tokens <= self.max_total_tokens
            let input_length = truncate.unwrap_or(self.max_input_length);

            Ok((inputs, input_length))
        }
    }

    #[instrument(skip_all)]
    pub(crate) async fn token_count(
        &self,
        inputs: String,
    ) -> Result<usize, ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender
                .send(((inputs, None), response_sender, Span::current()))
                .unwrap();

            // Await on response channel
            // Unwrap is safe here
            let (inputs, _, input_length) = response_receiver.await.unwrap()?;

            metrics::histogram!("tgi_request_input_length", input_length as f64);
            Ok(input_length)
        }
        // Return 0 to signify that we can't perform this op
        else {
            Ok(0)
        }
    }

    #[instrument(skip_all)]
    pub(crate) async fn tokenize(
        &self,
        inputs: String,
    ) -> Result<(Vec<u32>, usize), ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender
                .send(((inputs, None), response_sender, Span::current()))
                .unwrap();

            // Await on response channel
            // Unwrap is safe here
            let (inputs, encoding, input_length) = response_receiver.await.unwrap()?;

            metrics::histogram!("tgi_request_input_length", input_length as f64);
            Ok((encoding.get_ids().to_vec(), input_length))
        }
        // Return 0 to signify that we can't perform this op
        else {
            Ok((vec![], 0))
        }
    }

    /// Validate a payload and get the number of tokens in the input
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: EmbedRequest,
    ) -> Result<ValidEmbedRequest, ValidationError> {
        // Check if inputs is empty
        if request.inputs.is_empty() {
            return Err(EmptyInput);
        }

        // Validate inputs
        let (inputs, input_length) = self
            .validate_input(request.inputs, None)
            .await?;

        Ok(ValidEmbedRequest {
            inputs,
            input_length: input_length as u32,
        })
    }
}

/// Start tokenization workers
fn tokenizer_worker(tokenizer: Tokenizer, receiver: flume::Receiver<TokenizerRequest>) {
    // Loop over requests
    while let Ok(((inputs, truncate), response_tx, parent_span)) = receiver.recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(prepare_input(inputs, truncate, &tokenizer))
                .unwrap_or(())
        })
    }
}

/// Get input length and optionally truncate it
fn prepare_input(
    inputs: String,
    truncate: Option<usize>,
    tokenizer: &Tokenizer,
) -> Result<(String, Encoding, usize), ValidationError> {
    // Get the number of tokens in the input
    let mut encoding = tokenizer
        .encode(inputs.clone(), true)
        .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;

    // Optionally truncate
    let (inputs, encoding, input_length) = match truncate {
        // Truncate is some and < encoding length
        Some(truncate) if truncate < encoding.len() => {
            // truncate encoding and decode new inputs
            encoding.truncate(truncate, 0, TruncationDirection::Left);
            let inputs = tokenizer
                .decode(Vec::from(encoding.get_ids()), false)
                .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;
            (inputs, encoding, encoding.len())
        }
        // Nothing to do
        _ => (inputs, encoding, encoding.len()),
    };

    Ok((inputs, encoding, input_length))
}

type TokenizerRequest = (
    (String, Option<usize>),
    oneshot::Sender<Result<(String, Encoding, usize), ValidationError>>,
    Span,
);

#[derive(Debug)]
pub(crate) struct ValidEmbedRequest {
    pub inputs: String,
    pub input_length: u32,
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("`truncate` must be strictly positive and less than {0}. Given: {1}")]
    Truncate(usize, usize),
    #[error("`max_new_tokens` must be strictly positive")]
    NegativeMaxNewTokens,
    #[error("`inputs` must have less than {0} tokens. Given: {1}")]
    InputLength(usize, usize),
    #[error("`inputs` cannot be empty")]
    EmptyInput,
    #[error("tokenizer error {0}")]
    Tokenizer(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::get_tokenizer;

    #[tokio::test]
    async fn test_validation_input_length() {
        let tokenizer = Some(get_tokenizer().await);
        let max_input_length = 4;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_input_length,
        );

        match validation
            .validate_input("Hello this is a long message with too many tokens".to_string(), None)
            .await
        {
            Err(ValidationError::InputLength(4, 10)) => (),
            _ => panic!("Unexpected not input length"),
        }
    }

    #[tokio::test]
    async fn test_validation_token_count() {
        let tokenizer = Some(get_tokenizer().await);
        let max_input_length = 4;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_input_length,
        );

        let val1 = validation
            .token_count("Hello this is a long message with too many tokens".to_string());

        let val2 = validation
            .token_count("Hello this is a long message with too many tokens actually too many tokens".to_string());

        match val1.await {
            Ok(10) => (),
            Ok(token_count) => panic!("Unexpected token count: {}", token_count),
            Err(e) => panic!("Unexpected error: {}", e),
        }

        match val2.await {
            Ok(14) => (),
            Ok(token_count) => panic!("Unexpected token count: {}", token_count),
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
}
