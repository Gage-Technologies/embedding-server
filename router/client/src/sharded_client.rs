/// Multi shard Client
use crate::{Batch, Client, Execution, HealthResponse, ShardInfo};
use crate::{ClientError, Result};
use futures::future::join_all;
use tonic::transport::Uri;
use tracing::instrument;

#[derive(Debug, Clone)]
/// Text Generation Inference gRPC multi client
pub struct ShardedClient {
    clients: Vec<Client>,
}

impl ShardedClient {
    fn new(clients: Vec<Client>) -> Self {
        Self { clients }
    }

    /// Create a new ShardedClient from a master client. The master client will communicate with
    /// the other shards and returns all uris/unix sockets with the `service_discovery` gRPC method.
    async fn from_master_client(mut master_client: Client) -> Result<Self> {
        // Get all uris/unix sockets from the master client
        let uris = master_client.service_discovery().await?;
        let futures = uris.into_iter().map(Client::connect_uds);
        let clients: Result<Vec<Client>> = join_all(futures).await.into_iter().collect();
        Ok(Self::new(clients?))
    }

    /// Returns a client connected to the given uri
    pub async fn connect(uri: Uri) -> Result<Self> {
        let master_client = Client::connect(uri).await?;
        Self::from_master_client(master_client).await
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let master_client = Client::connect_uds(path).await?;
        Self::from_master_client(master_client).await
    }

    /// Get the model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<ShardInfo> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.info())
            .collect();
        join_all(futures).await.pop().unwrap()
    }

    /// GRPC health check
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.health())
            .collect();
        join_all(futures).await.pop().unwrap()
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.clear_cache(batch_id))
            .collect();
        join_all(futures).await.into_iter().collect()
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn embed(&mut self, batch: Batch) -> Result<Vec<Execution>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.embed(batch.clone())))
            .collect();
        let results: Result<Vec<Vec<Execution>>> =
            join_all(futures).await.into_iter().collect();
        merge_generations(results?)
    }
}

/// Merge generations from the different model shards
fn merge_generations(
    mut results: Vec<Vec<Execution>>,
) -> Result<Vec<Execution>> {
    let mut execs = results.pop().ok_or(ClientError::EmptyResults)?;

    for mut shard_exec in results.into_iter() {
        execs.append(&mut shard_exec);
    }
    Ok(execs)
}
