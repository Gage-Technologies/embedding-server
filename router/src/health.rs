use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use embedding_server_client::{
    Batch, Request, ShardedClient,
};

// Note: Request ids and batch ids cannot collide.
const LIVENESS_ID: u64 = u64::MAX;
const BATCH_ID: u64 = u64::MAX;

#[derive(Clone, Debug)]
pub(crate) struct Health {
    client: ShardedClient,
    generation_health: Arc<AtomicBool>,
}

impl Health {
    pub(crate) fn new(client: ShardedClient, generation_health: Arc<AtomicBool>) -> Self {
        Self {
            client,
            generation_health,
        }
    }

    pub(crate) async fn check(&mut self) -> bool {
        if self.generation_health.load(Ordering::SeqCst) {
            // Generation is healthy, we only check that the shards are answering gRPC calls
            self.client.health().await.is_ok()
        } else {
            // Generation is unhealthy or have not sent any generation request yet

            // Dummy batch of 1 token and 1 generated token
            let liveness_request = Request {
                id: LIVENESS_ID,
                inputs: "liveness".to_string(),
                truncate: 10,
            };
            let batch = Batch {
                id: BATCH_ID,
                requests: vec![liveness_request],
                size: 1,
            };
            // Skips the queue
            let value = self.client.embed(batch).await.is_ok();
            // Update generation health
            self.generation_health.store(value, Ordering::SeqCst);
            value
        }
    }
}
