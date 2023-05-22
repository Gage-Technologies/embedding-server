use std::time::{Duration, Instant};
use embedding_server_client::{
    Batch, ClientError, Request, ShardedClient,
};
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::sync::{broadcast, mpsc};

const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

#[derive(Debug, Clone)]
pub(crate) struct Embed {
    pub(crate) latency: Duration,
    pub(crate) throughput: f64,
}

#[derive(Debug)]
pub(crate) enum Message {
    Warmup,
    Embed(Embed),
    EndRun,
    EndBatch,
}

/// Benchmarking task
#[allow(clippy::too_many_arguments)]
pub(crate) async fn embed_task(
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    n_runs: usize,
    warmups: usize,
    client: ShardedClient,
    run_sender: mpsc::Sender<Result<Message, ClientError>>,
    mut shutdown_receiver: broadcast::Receiver<()>,
    _shutdown_guard_sender: mpsc::Sender<()>,
) {
    // End task if a message is received on shutdown_receiver
    // _shutdown_guard_sender will be dropped once the task is finished
    tokio::select! {
        res = embed_runs(tokenizer, batch_size, sequence_length, n_runs, warmups, client, run_sender.clone())  => {
            if let Err(err) = res {
                run_sender.send(Err(err)).await.unwrap_or(());
            }
        },
        _ = shutdown_receiver.recv() => {}
    }
}

/// Benchmark prefill/decode
#[allow(clippy::too_many_arguments)]
async fn embed_runs(
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    n_runs: usize,
    warmups: usize,
    mut client: ShardedClient,
    run_sender: mpsc::Sender<Result<Message, ClientError>>,
) -> Result<(), ClientError> {
    // Create a dummy sequence
    let sequence = create_sequence(sequence_length, tokenizer);

    for b in batch_size {
        // Warmups on batch size
        for _ in 0..warmups {
            embed(
                sequence.clone(),
                sequence_length,
                b,
                &mut client,
            )
            .await?;
            // Send warmup message
            run_sender.send(Ok(Message::Warmup)).await.unwrap_or(());
        }

        for _ in 0..n_runs {
            let e = embed(
                sequence.clone(),
                sequence_length,
                b,
                &mut client,
            )
            .await?;
            // Send prefill message
            run_sender
                .send(Ok(Message::Embed(e)))
                .await
                .unwrap_or(());

            // Send run ended message
            run_sender.send(Ok(Message::EndRun)).await.unwrap_or(());
        }
        // Batch ended
        run_sender.send(Ok(Message::EndBatch)).await.unwrap_or(());
    }
    Ok(())
}

// Run a prefill step
async fn embed(
    sequence: String,
    sequence_length: u32,
    batch_size: u32,
    client: &mut ShardedClient,
) -> Result<Embed, ClientError> {
    // Create requests
    let requests = (0..batch_size)
        .map(|id| Request {
            id: id.into(),
            inputs: sequence.clone(),
            truncate: sequence_length,
        })
        .collect();

    let batch = Batch {
        id: 0,
        requests,
        size: batch_size,
    };

    // Run prefill
    let start_time = Instant::now();
    client.embed(batch.clone()).await?;

    // Get latency
    let latency = start_time.elapsed();

    // Compute throughput from latency and batch size
    let throughput = batch_size as f64 / latency.as_secs_f64();

    let step = Embed {
        latency,
        throughput,
    };

    Ok(step)
}

/// Create a dummy sequence of the correct length
fn create_sequence(sequence_length: u32, tokenizer: Tokenizer) -> String {
    let lorem_ipsum_length = tokenizer.encode(LOREM_IPSUM, true).unwrap().len();
    // Repeat lorem ipsum to cover sequence length
    let string_sequence =
        LOREM_IPSUM.repeat((0..sequence_length).step_by(lorem_ipsum_length).len());
    // Encode sequence
    let mut encoding = tokenizer.encode(string_sequence, true).unwrap();
    // Truncate to sequence_length
    encoding.truncate(sequence_length as usize, 0, TruncationDirection::Left);
    // Decode
    tokenizer
        .decode(Vec::from(encoding.get_ids()), false)
        .unwrap()
}
