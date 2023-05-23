package embedding_server

// Request
// Mock of the embed request for the embedding server
type Request struct {
	Inputs string `json:"inputs"`
}

// Response
// Mock of the embed response for the embedding server
type Response struct {
	Embedding []float32 `json:"embedding"`
	Dim       int       `json:"dim"`
}
