package embedding_server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Client
// Wrapper client around the http api of the embedding server. Example usage:
//
//	client := embedding_server.NewClient(http://localhost:8080, nil, nil, time.Second * 30)
//	res, err := client.Embed("Hi there!")
//	if err != nil {
//	    panic(err)
//	}
//	fmt.Println("Embedding: ", res.Embedding)
//	fmt.Println("Dimension: ", res.Dim)
type Client struct {
	baseURL string
	client  *http.Client
	headers map[string]string
	cookies map[string]string
}

// NewClient
// Create a new client for the embedding server
func NewClient(baseURL string, headers map[string]string, cookies map[string]string, timeout time.Duration) *Client {
	// create custom http client with timeout
	client := &http.Client{
		Timeout: timeout,
	}

	return &Client{
		baseURL: baseURL,
		client:  client,
		headers: headers,
		cookies: cookies,
	}
}

// prepareRequest
// Prepare a request for the embedding server
func (c *Client) prepareRequest(req *http.Request) *http.Request {
	// set content type
	req.Header.Set("Content-Type", "application/json")

	// set the headers
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}

	// set the cookies
	for k, v := range c.cookies {
		req.AddCookie(&http.Cookie{
			Name:  k,
			Value: v,
		})
	}

	return req
}

// Info
// Get information about the embedding server
func (c *Client) Info() (*InfoResponse, error) {
    // create the http request
    httpReq, err := http.NewRequest(
        "GET",
        c.baseURL+"/info",
        nil,
    )

    // prepare the request
    httpReq = c.prepareRequest(httpReq)

    // execute the request
    res, err := c.client.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("failed to execute http request: %w", err)
    }

    // parse the response
    var response InfoResponse
    err = json.NewDecoder(res.Body).Decode(&response)
    if err != nil {
        return nil, fmt.Errorf("failed to parse response: %w", err)
    }

    return &response, nil
}

// TokenCount
// Get the number of tokens for the passed input using the
// native tokenizer of the model running on the embedding server.
func (c *Client) TokenCount(inputs string) (*TokenCountResponse, error) {
    // ensure input is not empty
    if inputs == "" {
        return nil, ErrEmptyInputs
    }

    // create the request
    req, err := json.Marshal(EmbedRequest{
        Inputs: inputs,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    // create the http request
    httpReq, err := http.NewRequest(
        "POST",
        c.baseURL+"/token-count",
		bytes.NewBuffer(req),
    )

    // prepare the request
    httpReq = c.prepareRequest(httpReq)

    // execute the request
    res, err := c.client.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("failed to execute http request: %w", err)
    }

    // parse the response
    var response TokenCountResponse
    err = json.NewDecoder(res.Body).Decode(&response)
    if err != nil {
        return nil, fmt.Errorf("failed to parse response: %w", err)
    }

    return &response, nil
}


// Tokenize
// Tokenize the content into the integer representations using the
// native tokenizer of the model running on the embedding server.
func (c *Client) Tokenize(inputs string) (*TokenizeResponse, error) {
    // ensure input is not empty
    if inputs == "" {
        return nil, ErrEmptyInputs
    }

    // create the request
    req, err := json.Marshal(EmbedRequest{
        Inputs: inputs,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    // create the http request
    httpReq, err := http.NewRequest(
        "POST",
        c.baseURL+"/tokenize",
		bytes.NewBuffer(req),
    )

    // prepare the request
    httpReq = c.prepareRequest(httpReq)

    // execute the request
    res, err := c.client.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("failed to execute http request: %w", err)
    }

    // parse the response
    var response TokenizeResponse
    err = json.NewDecoder(res.Body).Decode(&response)
    if err != nil {
        return nil, fmt.Errorf("failed to parse response: %w", err)
    }

    return &response, nil
}


// Embed
// Embed a string using the embedding server
func (c *Client) Embed(inputs string) (*EmbedResponse, error) {
	// ensure inputs is not empty
	if inputs == "" {
		return nil, ErrEmptyInputs
	}

	// create the request
	req, err := json.Marshal(EmbedRequest{
		Inputs: inputs,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// create the http request
	httpReq, err := http.NewRequest(
		"POST",
		c.baseURL+"/embed",
		bytes.NewBuffer(req),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}

	// prepare the request
	httpReq = c.prepareRequest(httpReq)

	// execute the request
	res, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to execute http request: %w", err)
	}

	// parse the response
	var response EmbedResponse
	err = json.NewDecoder(res.Body).Decode(&response)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &response, nil
}
