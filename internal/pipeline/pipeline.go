package pipeline

import (
	"context"

	types "github.com/inference-gateway/adk/types"
	sdk "github.com/inference-gateway/sdk"
)

// Stage represents a single step in the pipeline.
// It receives a tool call and returns zero or more messages, or an error.
type Stage func(ctx context.Context, call sdk.ChatCompletionMessageToolCall) ([]types.Message, error)

// Pipeline is a sequence of stages executed for each tool call.
type Pipeline struct {
	stages []Stage
}

// New creates a new pipeline with the given stages.
func New(stages ...Stage) *Pipeline {
	return &Pipeline{stages: stages}
}

// Run executes the pipeline sequentially for all tool calls.
// Results from all stages and all calls are accumulated and returned.
// If any stage returns an error, execution stops and the error is returned.
func (p *Pipeline) Run(ctx context.Context, calls []sdk.ChatCompletionMessageToolCall) ([]types.Message, error) {
	var results []types.Message

	for _, call := range calls {
		for _, stage := range p.stages {
			msgs, err := stage(ctx, call)
			if err != nil {
				return results, err
			}
			results = append(results, msgs...)
		}
	}

	return results, nil
}
