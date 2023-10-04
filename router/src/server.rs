use crate::health::Health;
/// HTTP Server logic
use crate::infer::{InferError};
use crate::{
    CompatEmbedRequest, EmbedRequest, EmbedResponse, TokenCountResponse, ErrorResponse, HubModelInfo, Infer, Info,
    Validation, TokenizeResponse
};
use axum::extract::Extension;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use embedding_server_client::{ShardInfo, ShardedClient};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::signal;
use tokio::time::Instant;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{instrument};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

// TODO: so we keep this???
/// Generate tokens if `stream == false` or a stream of token if `stream == true`
// #[utoipa::path(
//     post,
//     tag = "Embedding Server Inference",
//     path = "/",
//     request_body = CompatEmbedRequest,
//     responses(
//         (status = 200, description = "Generated Text",
//             content(
//                 ("application/json" = EmbedResponse),
//                 ("text/event-stream" = StreamResponse),
//             )),
//         (status = 424, description = "Generation Error", body = ErrorResponse,
//             example = json ! ({"error": "Request failed during generation"})),
//         (status = 429, description = "Model is overloaded", body = ErrorResponse,
//             example = json ! ({"error": "Model is overloaded"})),
//         (status = 422, description = "Input validation error", body = ErrorResponse,
//             example = json ! ({"error": "Input validation error"})),
//         (status = 500, description = "Incomplete generation", body = ErrorResponse,
//             example = json ! ({"error": "Incomplete generation"})),
//     )
// )]
// #[instrument(skip(infer))]
// async fn compat_generate(
//     default_return_full_text: Extension<bool>,
//     infer: Extension<Infer>,
//     req: Json<CompatEmbedRequest>,
// ) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
//     let mut req = req.0;

//     // default return_full_text given the pipeline_tag
//     if req.parameters.return_full_text.is_none() {
//         req.parameters.return_full_text = Some(default_return_full_text.0)
//     }

//     // switch on stream
//     if req.stream {
//         Ok(generate_stream(infer, Json(req.into()))
//             .await
//             .into_response())
//     } else {
//         let (headers, generation) = generate(infer, Json(req.into())).await?;
//         // wrap generation inside a Vec to match api-inference
//         Ok((headers, Json(vec![generation.0])).into_response())
//     }
// }

/// Embedding Server Inference endpoint info
#[utoipa::path(
    get,
    tag = "Embedding Server Inference",
    path = "/info",
    responses((status = 200, description = "Served model info", body = Info))
)]
#[instrument]
async fn get_model_info(info: Extension<Info>) -> Json<Info> {
    Json(info.0)
}

#[utoipa::path(
    get,
    tag = "Embedding Server Inference",
    path = "/health",
    responses(
        (status = 200, description = "Everything is working fine"),
        (status = 503, description = "Embedding Server Inference is down", body = ErrorResponse,
            example = json ! ({"error": "unhealthy", "error_type": "healthcheck"})),
    )
)]
#[instrument(skip(health))]
/// Health check method
async fn health(mut health: Extension<Health>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match health.check().await {
        true => Ok(()),
        false => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "unhealthy".to_string(),
                error_type: "healthcheck".to_string(),
            }),
        )),
    }
}

/// Generate tokens
#[utoipa::path(
    post,
    tag = "Embedding Server Inference",
    path = "/embed",
    request_body = EmbedRequest,
    responses(
        (status = 200, description = "Embedded Text", body = EmbedResponse),
        (status = 424, description = "Embedding Error", body = ErrorResponse,
            example = json ! ({"error": "Request failed during generation"})),
        (status = 429, description = "Model is overloaded", body = ErrorResponse,
            example = json ! ({"error": "Model is overloaded"})),
        (status = 422, description = "Input validation error", body = ErrorResponse,
            example = json ! ({"error": "Input validation error"})),
        (status = 500, description = "Incomplete generation", body = ErrorResponse,
            example = json ! ({"error": "Incomplete generation"})),
    )
)]
#[instrument(
    skip(infer),
    fields(
        total_time,
        validation_time,
        queue_time,
        inference_time,
        time_per_token,
        seed,
    )
)]
async fn embed(
    infer: Extension<Infer>,
    req: Json<EmbedRequest>,
) -> Result<(HeaderMap, Json<EmbedResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("tgi_request_count");

    let compute_characters = req.0.inputs.chars().count();

    // Inference
    let response = infer.embed(req.0).await?;

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("tgi_request_success");
    metrics::histogram!("tgi_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "tgi_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("tgi_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "tgi_request_inference_duration",
        inference_time.as_secs_f64()
    );

    // Send response
    let res = EmbedResponse {
        embedding: response.embedding.embedding,
        dim: response.embedding.dim,
    };
    Ok((headers, Json(res)))
}

/// Count tokens
#[utoipa::path(
    post,
    tag = "Embedding Server Inference",
    path = "/token_count",
    request_body = EmbedRequest,
    responses(
        (status = 200, description = "Token count", body = TokenCountResponse),
        (status = 424, description = "Embedding Error", body = ErrorResponse,
            example = json ! ({"error": "Request failed during generation"})),
        (status = 429, description = "Model is overloaded", body = ErrorResponse,
            example = json ! ({"error": "Model is overloaded"})),
        (status = 422, description = "Input validation error", body = ErrorResponse,
            example = json ! ({"error": "Input validation error"})),
        (status = 500, description = "Incomplete generation", body = ErrorResponse,
            example = json ! ({"error": "Incomplete generation"})),
    )
)]
async fn token_count(
    infer: Extension<Infer>,
    req: Json<EmbedRequest>,
) -> Result<(HeaderMap, Json<TokenCountResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("tgi_tc_request_count");

    let compute_characters = req.0.inputs.chars().count();

    // Inference
    let response = infer.token_count(req.0).await?;

    // Timings
    let total_time = start_time.elapsed();

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("tgi_tc_request_success");
    metrics::histogram!("tgi_tc_request_duration", total_time.as_secs_f64());

    // Send response
    let res = TokenCountResponse {
        count: response.count,
    };
    Ok((headers, Json(res)))
}

/// Tokenize content
#[utoipa::path(
post,
tag = "Embedding Server Inference",
path = "/tokenize",
request_body = EmbedRequest,
responses(
(status = 200, description = "Tokenized content", body = TokenCountResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
async fn tokenize(
    infer: Extension<Infer>,
    req: Json<EmbedRequest>,
) -> Result<(HeaderMap, Json<TokenizeResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("tgi_tc_request_tokenize");

    let compute_characters = req.0.inputs.chars().count();

    // Inference
    let response = infer.tokenize(req.0).await?;

    // Timings
    let total_time = start_time.elapsed();

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("tgi_tc_request_success");
    metrics::histogram!("tgi_tc_request_duration", total_time.as_secs_f64());

    // Send response
    let res = TokenizeResponse {
        tokens: response.tokens,
        count: response.count,
    };
    Ok((headers, Json(res)))
}

/// Prometheus metrics scrape endpoint
#[utoipa::path(
    get,
    tag = "Embedding Server Inference",
    path = "/metrics",
    responses((status = 200, description = "Prometheus Metrics", body = String))
)]
async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_info: HubModelInfo,
    shard_info: ShardInfo,
    compat_return_full_text: bool,
    max_concurrent_requests: usize,
    max_input_length: usize,
    waiting_served_ratio: f32,
    max_batch_total_tokens: u32,
    client: ShardedClient,
    tokenizer: Option<Tokenizer>,
    validation_workers: usize,
    addr: SocketAddr,
    allow_origin: Option<AllowOrigin>,
) {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
        paths(
            get_model_info,
            // compat_generate,
            embed,
            metrics,
        ),
        components(
            schemas(
                Info,
                CompatEmbedRequest,
                EmbedRequest,
                EmbedResponse,
                TokenCountResponse,
                ErrorResponse,
            )
        ),
        tags(
            (name = "Embedding Server Inference", description = "Hugging Face Embedding Server Inference API")
        ),
        info(
            title = "Embedding Server Inference",
            license(
                name = "Apache 2.0",
                url = "https://www.apache.org/licenses/LICENSE-2.0"
            )
        )
    )]
    struct ApiDoc;

    // Create state
    let validation = Validation::new(validation_workers, tokenizer, max_input_length);
    let generation_health = Arc::new(AtomicBool::new(false));
    let health_ext = Health::new(client.clone(), generation_health.clone());
    let infer = Infer::new(
        client,
        validation,
        max_batch_total_tokens,
        max_concurrent_requests,
        generation_health,
    );

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.0001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }
    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("tgi_request_input_length"));
    let input_length_buckets: Vec<f64> = (0..100)
        .map(|x| (max_input_length as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("tgi_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..1024).map(|x| (x + 1) as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap();
    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    // Endpoint info
    let info = Info {
        model_id: model_info.model_id,
        model_sha: model_info.sha,
        model_dtype: shard_info.dtype,
        model_device_type: shard_info.device_type,
        model_pipeline_tag: model_info.pipeline_tag,
        model_dim: shard_info.dim as usize,
        max_concurrent_requests,
        max_input_length,
        waiting_served_ratio,
        max_batch_total_tokens,
        validation_workers,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
    };

    // Create router
    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
        // Base routes
        // .route("/", post(compat_generate))
        .route("/info", get(get_model_info))
        .route("/embed", post(embed))
        .route("/token-count", post(token_count))
        // AWS Sagemaker route
        // .route("/invocations", post(compat_generate))
        // Base Health route
        .route("/health", get(health))
        // Inference API health route
        .route("/", get(health))
        // AWS Sagemaker health route
        .route("/ping", get(health))
        // Prometheus metrics route
        .route("/metrics", get(metrics))
        .layer(Extension(info))
        .layer(Extension(health_ext))
        .layer(Extension(compat_return_full_text))
        .layer(Extension(infer))
        .layer(Extension(prom_handle))
        .layer(opentelemetry_tracing_layer())
        .layer(cors_layer);

    // Run server
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        // Wait until all requests are finished to shut down
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
    opentelemetry::global::shutdown_tracer_provider();
}

/// Convert to Axum supported formats
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        let status_code = match err {
            InferError::ExecutionError(_) => StatusCode::FAILED_DEPENDENCY,
            InferError::Overloaded(_) => StatusCode::TOO_MANY_REQUESTS,
            InferError::ValidationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::MissingEmbedding => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (
            status_code,
            Json(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            }),
        )
    }
}

impl From<InferError> for Event {
    fn from(err: InferError) -> Self {
        Event::default()
            .json_data(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            })
            .unwrap()
    }
}
