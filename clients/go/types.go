package embedding_server

// EmbedRequest
// Mock of the embed request for the embedding server
type EmbedRequest struct {
	Inputs string `json:"inputs"`
}

// EmbedResponse
// Mock of the embed response for the embedding server
type EmbedResponse struct {
	Embedding []float32 `json:"embedding"`
	Dim       int       `json:"dim"`
}

// InfoResponse
// Mock of the info response for the embedding server
type InfoResponse struct {
    ModelID            string  `json:"model_id"`
    ModelSHA           string  `json:"model_sha"`
    ModelDtype         string  `json:"model_dtype"`
    ModelDeviceType    string  `json:"model_device_type"`
    ModelPipelineTag   string  `json:"model_pipeline_tag"`
    ModelDim           int     `json:"model_dim"`
    MaxConcurrentRequests int `json:"max_concurrent_requests"`
    MaxInputLength     int     `json:"max_input_length"`
    WaitingServedRatio float32 `json:"waiting_served_ratio"`
    MaxBatchTotalTokens int    `json:"max_batch_total_tokens"`
    ValidationWorkers  int     `json:"validation_workers"`
    Version            string  `json:"version"`
    Sha                string  `json:"sha"`
    DockerLabel        string  `json:"docker_label"`
}

// TokenCountResponse
// Mock of the token count response for the embedding server
type TokenCountResponse struct {
    Count int `json:"count"`
}

// TokenCountResponse
// Mock of the tokenize response for the embedding server
type TokenizeResponse struct {
    Tokens []uint `json:"tokens"`
    Count int `json:"count"`
}