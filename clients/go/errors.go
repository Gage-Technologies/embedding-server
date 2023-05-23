package embedding_server

import "errors"

var (
	// ErrEmptyInputs is returned when the inputs are empty
	ErrEmptyInputs = errors.New("inputs cannot be empty")
)
