/// Batching and inference logic
use crate::validation::{Validation, ValidationError};
use crate::{Entry, Queue};
use crate::{EmbedRequest};
use flume::SendError;
use nohash_hasher::IntMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use embedding_server_client::{
    Batch, ClientError, Embedding, Execution, ShardedClient,
};
use thiserror::Error;
use tokio::sync::{Notify, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tracing::{info_span, instrument, Instrument, Span};

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// Validation
    validation: Validation,
    /// Request queue
    queue: Queue,
    /// Shared state
    shared: Arc<Shared>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
}

/// Infer shared state
struct Shared {
    /// Batching background Tokio task notifier
    batching_task: Notify,
}

impl Infer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        client: ShardedClient,
        validation: Validation,
        max_batch_total_tokens: u32,
        max_concurrent_requests: usize,
        system_health: Arc<AtomicBool>,
    ) -> Self {
        // Infer shared state
        let queue = Queue::new();
        let shared = Arc::new(Shared {
            batching_task: Notify::new(),
        });

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client,
            max_batch_total_tokens,
            queue.clone(),
            shared.clone(),
            system_health,
        ));

        // Inference limit with a semaphore
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        Self {
            validation,
            queue,
            shared,
            limit_concurrent_requests: semaphore,
        }
    }

    /// Add a new request to the queue and return a stream of InferStreamResponse
    #[instrument(skip(self))]
    pub(crate) async fn embed(
        &self,
        request: EmbedRequest,
    ) -> Result<InferResponse, InferError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("tgi_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        // Validate request
        let valid_request = self.validation.validate(request).await.map_err(|err| {
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            err
        })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = flume::unbounded();

        // Append the request to the queue
        self.queue.append(Entry {
            request: valid_request,
            response_tx,
            span: Span::current(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
        });

        // Notify the background task that we have a new entry in the queue that needs
        // to be batched
        self.shared.batching_task.notify_one();

        // wait for the response
        match response_rx.recv_async().await {
            Ok(response) => {
                // Release the semaphore permit
                drop(permit);
                Ok(response?)
            }
            Err(err) => {
                // Release the semaphore permit
                drop(permit);
                Err(InferError::ExecutionError(err.to_string()))
            }
        }
    }

    // /// Add a new request to the queue and return a InferResponse
    // #[instrument(skip(self))]
    // pub(crate) async fn generate(
    //     &self,
    //     request: EmbedRequest,
    // ) -> Result<InferResponse, InferError> {
    //     // Create stream and keep semaphore permit as long as generate lives
    //     let (_permit, mut stream) = self.embed(request).await?;
    //
    //     // Return values
    //     let mut result_prefill = Vec::new();
    //     let mut result_tokens = Vec::new();
    //     let mut result_generated_text = None;
    //     let mut result_start = None;
    //     let mut result_queued = None;
    //
    //     // Iterate on stream
    //     while let Some(response) = stream.next().await {
    //         match response? {
    //             // Add prefill tokens
    //             InferStreamResponse::Prefill(tokens) => {
    //                 // Create Token objects
    //                 // We do that here instead of in the Python code as Rust for loops are faster
    //                 result_prefill = tokens
    //                     .ids
    //                     .into_iter()
    //                     .zip(tokens.logprobs.into_iter())
    //                     .zip(tokens.texts.into_iter())
    //                     .map(|((id, logprob), text)| PrefillToken { id, text, logprob })
    //                     .collect();
    //             }
    //             // Push last token
    //             InferStreamResponse::Token(token) => result_tokens.push(token),
    //             // Final message
    //             // Set return values
    //             InferStreamResponse::End {
    //                 token,
    //                 generated_text,
    //                 start,
    //                 queued,
    //             } => {
    //                 result_tokens.push(token);
    //                 result_generated_text = Some(generated_text);
    //                 result_start = Some(start);
    //                 result_queued = Some(queued)
    //             }
    //         }
    //     }
    //
    //     // Check that we received a `InferStreamResponse::End` message
    //     if let (Some(generated_text), Some(queued), Some(start)) =
    //         (result_generated_text, result_queued, result_start)
    //     {
    //         Ok(InferResponse {
    //             prefill: result_prefill,
    //             tokens: result_tokens,
    //             generated_text,
    //             queued,
    //             start,
    //         })
    //     } else {
    //         let err = InferError::IncompleteGeneration;
    //         metrics::increment_counter!("tgi_request_failure", "err" => "incomplete");
    //         tracing::error!("{err}");
    //         Err(err)
    //     }
    // }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
async fn batching_task(
    mut client: ShardedClient,
    max_batch_total_tokens: u32,
    queue: Queue,
    shared: Arc<Shared>,
    generation_health: Arc<AtomicBool>,
) {
    // Infinite loop
    loop {
        // Wait for a notification from the Infer struct
        shared.batching_task.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the queue
        while let Some((mut entries, batch, span)) =
            queue.next_batch(None, max_batch_total_tokens).await
        {
            embed(&mut client, batch, &mut entries, &generation_health)
                .instrument(span)
                .await;

            metrics::gauge!("tgi_batch_current_size", 0.0);
            metrics::gauge!("tgi_batch_current_max_tokens", 0.0);
        }
    }
}

#[instrument(skip_all)]
async fn embed(
    client: &mut ShardedClient,
    batch: Batch,
    entries: &mut IntMap<u64, Entry>,
    system_health: &Arc<AtomicBool>,
) {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::increment_counter!("tgi_batch_inference_count", "method" => "prefill");

    match client.embed(batch).await {
        Ok(execs) => {
            // Update health
            system_health.store(true, Ordering::SeqCst);
            // Send embeddings back to the callers
            return_embeddings(execs, entries);

            metrics::histogram!("tgi_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "prefill");
            metrics::increment_counter!("tgi_batch_inference_success", "method" => "prefill");
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            // Update health
            system_health.store(false, Ordering::SeqCst);
            let _ = client.clear_cache(Some(batch_id)).await;
            send_errors(err, entries);
            metrics::increment_counter!("tgi_batch_inference_failure", "method" => "prefill");
        }
    }
}

/// Pipe embeddings back to the callers
#[instrument(skip_all)]
fn return_embeddings(execs: Vec<Execution>, entries: &mut IntMap<u64, Entry>) {
    execs.into_iter().for_each(|exec| {
        let id = exec.request_id;
        // Get entry
        // We can `expect` here as the request id should always be in the entries
        let entry = entries
            .get(&id)
            .expect("ID not found in entries. This is a bug.");

        // Create and enter a span to link this function back to the entry
        let _span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_exec", exec = ?exec).entered();
        // Send embedding responses back to the infer task
        // If the receive an error from the Flume channel, it means that the client dropped the request
        send_responses(exec, entry).map_err(|err| {
            metrics::increment_counter!("tgi_request_failure", "err" => "dropped");
            err
        });
        entries.remove(&id).expect("ID not found in entries. This is a bug.");
    });
}

/// Send responses through the `entry` response channel
fn send_responses(
    exec: Execution,
    entry: &Entry,
) -> Result<(), SendError<Result<InferResponse, InferError>>> {
    if let Some(embeddings) = exec.embedding {
        // Send message
        entry.response_tx.send(Ok(InferResponse{
            embedding: embeddings,
            // these are overridden on the receiver side
            queued: Instant::now(),
            start: Instant::now(),
        }))?;
    } else {
        // Send message
        entry
            .response_tx
            .send(Err(InferError::MissingEmbedding))?;
    }
    Ok(())
}

/// Send errors to Infer for all `entries`
#[instrument(skip_all)]
fn send_errors(error: ClientError, entries: &mut IntMap<u64, Entry>) {
    entries.drain().for_each(|(_, entry)| {
        // Create and enter a span to link this function back to the entry
        let _send_error_span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_error").entered();
        let err = InferError::ExecutionError(error.to_string());
        metrics::increment_counter!("tgi_request_failure", "err" => "generation");
        tracing::error!("{err}");

        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry
            .response_tx
            .send(Err(err))
            .unwrap_or(());
    });
}

#[derive(Debug)]
pub(crate) struct InferResponse {
    pub(crate) embedding: Embedding,
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during execution: {0}")]
    ExecutionError(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Input validation error: {0}")]
    ValidationError(#[from] ValidationError),
    #[error("Missing embedding")]
    MissingEmbedding,
}

impl InferError {
    pub(crate) fn error_type(&self) -> &str {
        match self {
            InferError::ExecutionError(_) => "execution",
            InferError::Overloaded(_) => "overloaded",
            InferError::ValidationError(_) => "validation",
            InferError::MissingEmbedding => "missing_embedding",
        }
    }
}
