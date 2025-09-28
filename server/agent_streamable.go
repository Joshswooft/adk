package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	cloudevents "github.com/cloudevents/sdk-go/v2"
	"github.com/inference-gateway/adk/internal/pipeline"
	types "github.com/inference-gateway/adk/types"
	sdk "github.com/inference-gateway/sdk"
	zap "go.uber.org/zap"
)

type EventPublisher interface {
	Publish(ctx context.Context, event cloudevents.Event) error
}

type ChanEventPublisher struct {
	Out chan<- cloudevents.Event
}

func (p *ChanEventPublisher) Publish(ctx context.Context, e cloudevents.Event) error {
	select {
	case p.Out <- e:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// RunWithStream processes a conversation and returns a streaming response with iterative tool calling support
// TODO: add more tests which use the callbacks
func (a *OpenAICompatibleAgentImpl) RunWithStream(ctx context.Context, messages []types.Message) (<-chan cloudevents.Event, error) {
	if a.llmClient == nil {
		return nil, fmt.Errorf("no LLM client configured for agent")
	}

	var tools []sdk.ChatCompletionTool
	if a.toolBox != nil {
		tools = a.toolBox.GetTools()
	}

	outputChan := make(chan cloudevents.Event, 100)

	// Create callback context for this streaming invocation
	var callbackContext *CallbackContext
	if a.callbackExecutor != nil {
		callbackContext = &CallbackContext{
			AgentName:    "agent", // TODO: Get actual agent name from context
			InvocationID: fmt.Sprintf("streaming-invocation-%d", time.Now().UnixNano()),
			Logger:       a.logger,
		}
		// Execute BeforeAgent callback if configured
		skipResult := a.callbackExecutor.ExecuteBeforeAgent(ctx, callbackContext)
		if skipResult != nil {
			// Callback returned a message, skip streaming execution
			go func() {
				defer close(outputChan)
				// Send the skip result as a completion event
				completionEvent := types.NewMessageEvent("adk.agent.stream.completed", skipResult.MessageID, skipResult, nil)
				select {
				case outputChan <- completionEvent:
				case <-ctx.Done():
				}
			}()
			return outputChan, nil
		}
	}

	go func() {
		defer close(outputChan)

		currentMessages := make([]types.Message, len(messages))
		copy(currentMessages, messages)

		for iteration := 1; iteration <= a.config.MaxChatCompletionIterations; iteration++ {
			a.logger.Debug("starting streaming iteration",
				zap.Int("iteration", iteration),
				zap.Int("message_count", len(currentMessages)))

			sdkMessages, err := a.converter.ConvertToSDK(currentMessages)
			if err != nil {
				a.logger.Error("failed to convert messages to SDK format", zap.Error(err))
				return
			}

			if a.config != nil && a.config.SystemPrompt != "" {
				systemMessage := sdk.Message{
					Role:    sdk.System,
					Content: a.config.SystemPrompt,
				}
				sdkMessages = append([]sdk.Message{systemMessage}, sdkMessages...)
			}

			// Execute BeforeModel callback if configured
			var streamSkipped bool
			var preGeneratedResponse *types.Message
			if a.callbackExecutor != nil && callbackContext != nil {
				// Convert conversation to LLM request format for callback
				a2aMessages := make([]types.Message, len(currentMessages))
				copy(a2aMessages, currentMessages)

				llmRequest := &LLMRequest{
					Contents: a2aMessages,
					Config:   &LLMConfig{
						// TODO: Add system instruction from config
					},
				}

				llmResponse := a.callbackExecutor.ExecuteBeforeModel(ctx, callbackContext, llmRequest)
				if llmResponse != nil {
					// Callback returned a response, use it instead of calling LLM
					streamSkipped = true
					preGeneratedResponse = llmResponse.Content
					a.logger.Debug("BeforeModel callback provided response, skipping LLM streaming call", zap.Int("iteration", iteration))
				}
			}

			var streamResponseChan <-chan *sdk.CreateChatCompletionStreamResponse
			var streamErrorChan <-chan error
			if !streamSkipped {
				streamResponseChan, streamErrorChan = a.llmClient.CreateStreamingChatCompletion(ctx, sdkMessages, tools...)
			}

			var fullContent string
			toolCallAccumulator := make(map[string]*sdk.ChatCompletionMessageToolCall)
			var assistantMessage *types.Message
			var toolResultMessages []types.Message
			toolResults := make(map[string]*types.Message)

			// Handle pre-generated response from callback
			if streamSkipped && preGeneratedResponse != nil {
				assistantMessage = preGeneratedResponse
				// Send the pre-generated response as a delta event
				for _, part := range assistantMessage.Parts {
					if partMap, ok := part.(map[string]any); ok {
						if textVal, hasText := partMap["text"].(string); hasText {
							chunkMessage := types.NewAssistantMessage(
								fmt.Sprintf("callback-chunk-%d", iteration),
								[]types.Part{types.NewTextPart(textVal)},
							)
							select {
							case outputChan <- types.NewDeltaEvent(chunkMessage):
							case <-ctx.Done():
								return
							}
						}
					}
				}
			} else {
				// Normal streaming execution
				streaming := true
				for streaming {
					select {
					case <-ctx.Done():
						a.logger.Info("streaming context cancelled, preserving partial state",
							zap.Int("iteration", iteration),
							zap.Bool("has_assistant_message", assistantMessage != nil),
							zap.Int("content_length", len(fullContent)),
							zap.Int("tool_result_count", len(toolResultMessages)),
							zap.Int("pending_tool_calls", len(toolCallAccumulator)))

						if assistantMessage != nil {
							iterationEvent := types.NewIterationCompletedEvent(iteration, "streaming-task", assistantMessage)
							select {
							case outputChan <- iterationEvent:
							case <-time.After(100 * time.Millisecond):
							}
						}

						interruptedTask := &types.Task{
							ID:        fmt.Sprintf("interrupted-%d", iteration),
							ContextID: fmt.Sprintf("streaming-task-%d", iteration),
							Status:    types.TaskStatus{State: types.TaskStateWorking},
						}
						interruptMessage := types.NewStreamingStatusMessage(
							fmt.Sprintf("task-interrupted-%d", iteration),
							// indicates an interrupt: https://a2a-protocol.org/dev/specification/#63-taskstate-enum
							types.TaskStateInputRequired,
							// "interrupted",
							map[string]any{
								"reason": "context_cancelled",
								"task":   interruptedTask,
							},
						)
						select {
						case outputChan <- types.NewMessageEvent("adk.agent.task.interrupted", interruptMessage.MessageID, interruptMessage, nil):
						default:
						}
						return

					case streamErr := <-streamErrorChan:
						if streamErr != nil {
							a.logger.Error("streaming failed", zap.Error(streamErr))

							errorMessage := types.NewStreamingStatusMessage(
								fmt.Sprintf("streaming-error-%d", iteration),
								types.TaskStateFailed,
								map[string]any{
									"error":     streamErr.Error(),
									"iteration": iteration,
								},
							)
							select {
							case outputChan <- types.NewMessageEvent("adk.agent.stream.failed", errorMessage.MessageID, errorMessage, nil):
							default:
							}
							return
						}
						streaming = false

					case streamResp, ok := <-streamResponseChan:
						if !ok {
							streaming = false
							break
						}

						if streamResp == nil || len(streamResp.Choices) == 0 {
							continue
						}

						choice := streamResp.Choices[0]

						if choice.Delta.Content != "" {
							fullContent += choice.Delta.Content

							chunkMessage := types.NewAssistantMessage(
								fmt.Sprintf("chunk-%d-%d", iteration, len(fullContent)),
								[]types.Part{types.NewTextPart(choice.Delta.Content)},
							)

							select {
							case outputChan <- types.NewDeltaEvent(chunkMessage):
							case <-ctx.Done():
								return
							}
						}

						for _, toolCallChunk := range choice.Delta.ToolCalls {
							key := fmt.Sprintf("%d", toolCallChunk.Index)

							if toolCallAccumulator[key] == nil {
								toolCallAccumulator[key] = &sdk.ChatCompletionMessageToolCall{
									Type:     "function",
									Function: sdk.ChatCompletionMessageToolCallFunction{},
								}
							}

							toolCall := toolCallAccumulator[key]
							if toolCallChunk.ID != "" {
								toolCall.Id = toolCallChunk.ID
							}
							if toolCallChunk.Function.Name != "" {
								toolCall.Function.Name = toolCallChunk.Function.Name
							}
							if toolCallChunk.Function.Arguments != "" {
								if toolCall.Function.Arguments == "" {
									toolCall.Function.Arguments = toolCallChunk.Function.Arguments
								} else if !isCompleteJSON(toolCall.Function.Arguments) {
									toolCall.Function.Arguments += toolCallChunk.Function.Arguments
								}
							}
						}

						if choice.FinishReason != "" {
							assistantMessage = types.NewAssistantMessage(
								fmt.Sprintf("assistant-stream-%d", iteration),
								make([]types.Part, 0),
							)

							if fullContent != "" {
								assistantMessage.Parts = append(assistantMessage.Parts, map[string]any{
									"kind": "text",
									"text": fullContent,
								})
							}
							streaming = false
						}
					}
				}
			} // End of normal streaming execution

			// Execute AfterModel callback if configured and we have an assistant message
			if a.callbackExecutor != nil && callbackContext != nil && assistantMessage != nil {
				originalResponse := &LLMResponse{
					Content: assistantMessage,
				}
				modifiedResponse := a.callbackExecutor.ExecuteAfterModel(ctx, callbackContext, originalResponse)
				if modifiedResponse != nil {
					// Use modified response
					assistantMessage = modifiedResponse.Content
					a.logger.Debug("AfterModel callback modified streaming response", zap.Int("iteration", iteration))
				}
			}

			if len(toolCallAccumulator) > 0 && a.toolBox != nil {
				for key, toolCall := range toolCallAccumulator {
					a.logger.Debug("tool call accumulator",
						zap.String("key", key),
						zap.String("id", toolCall.Id),
						zap.String("name", toolCall.Function.Name),
						zap.String("arguments", toolCall.Function.Arguments))
				}

				toolCalls := make([]sdk.ChatCompletionMessageToolCall, 0, len(toolCallAccumulator))
				for _, toolCall := range toolCallAccumulator {
					toolCalls = append(toolCalls, *toolCall)
				}

				assistantMessage.Parts = append(assistantMessage.Parts, map[string]any{
					"kind": "data",
					"data": map[string]any{
						"tool_calls": toolCalls,
					},
				})

				currentMessages = append(currentMessages, *assistantMessage)
				iterationEvent := types.NewIterationCompletedEvent(iteration, "streaming-task", assistantMessage)
				select {
				case outputChan <- iterationEvent:
				case <-ctx.Done():
					return
				}

				// toolResultMessages = a.executeToolCallsWithEvents(ctx, toolCalls, outputChan)
				callbackContext = &CallbackContext{
					AgentName:    "agent", // TODO: Get actual agent name from context
					InvocationID: fmt.Sprintf("streaming-invocation-%d", time.Now().UnixNano()),
					Logger:       a.logger,
				}
				// TODO: handle error?!
				toolResultMessages, _ := a.executeTools(ctx, toolCalls, callbackContext)
				for _, toolResult := range toolResultMessages {
					for _, part := range toolResult.Parts {
						if partMap, ok := part.(map[string]any); ok {
							if dataMap, exists := partMap["data"].(map[string]any); exists {
								if toolCallID, idExists := dataMap["tool_call_id"].(string); idExists {
									toolResults[toolCallID] = &toolResult
									break
								}
							}
						}
					}
				}

			} else {
				currentMessages = append(currentMessages, *assistantMessage)
				iterationEvent := types.NewIterationCompletedEvent(iteration, "streaming-task", assistantMessage)
				select {
				case outputChan <- iterationEvent:
				case <-ctx.Done():
					return
				}

			}

			// FIXME: where does this go?
			// streaming = false

			// FIXME section
			if len(toolResultMessages) > 0 {
				currentMessages = append(currentMessages, toolResultMessages...)
				a.logger.Debug("persisted tool result messages",
					zap.Int("iteration", iteration),
					zap.Int("tool_result_count", len(toolResultMessages)))
			}

			if len(toolResultMessages) > 0 {
				lastToolMessage := toolResultMessages[len(toolResultMessages)-1]
				if lastToolMessage.Kind == "input_required" {
					a.logger.Debug("streaming completed - input required from user",
						zap.Int("iteration", iteration),
						zap.Int("final_message_count", len(currentMessages)))
					return
				}
			}

			if assistantMessage != nil && len(toolResultMessages) == 0 {
				// Execute AfterAgent callback if configured before completion
				if a.callbackExecutor != nil && callbackContext != nil {
					modifiedResponse := a.callbackExecutor.ExecuteAfterAgent(ctx, callbackContext, assistantMessage)
					if modifiedResponse != nil {
						assistantMessage = modifiedResponse
						a.logger.Debug("AfterAgent callback modified final streaming response", zap.Int("iteration", iteration))
						// Send the modified response as a final event
						finalEvent := types.NewMessageEvent("adk.agent.stream.modified", assistantMessage.MessageID, assistantMessage, nil)
						select {
						case outputChan <- finalEvent:
						case <-ctx.Done():
						}
					}
				}

				a.logger.Debug("streaming completed - no tool calls executed",
					zap.Int("iteration", iteration),
					zap.Int("final_message_count", len(currentMessages)),
					zap.Bool("has_assistant_message", assistantMessage != nil))
				return
			}

			a.logger.Debug("tool calls executed, continuing to next iteration",
				zap.Int("iteration", iteration),
				zap.Int("message_count", len(currentMessages)),
				zap.Int("tool_results_count", len(toolResultMessages)),
				zap.Int("unique_tool_calls", len(toolResults)))

			// FIXME: end section

		}

		a.logger.Warn("max streaming iterations reached", zap.Int("max_iterations", a.config.MaxChatCompletionIterations))
	}()

	return outputChan, nil
}

// TODO: refactor me
// executeToolCallsWithEvents executes tool calls and emits events, returning tool result messages
// Now supports callbacks for tool execution
func (a *OpenAICompatibleAgentImpl) executeToolCallsWithEvents(ctx context.Context, toolCalls []sdk.ChatCompletionMessageToolCall, outputChan chan<- cloudevents.Event) []types.Message {
	// Create callback context for tool execution
	var callbackContext *CallbackContext
	if a.callbackExecutor != nil {
		callbackContext = &CallbackContext{
			AgentName:    "agent", // TODO: Get actual agent name from context
			InvocationID: fmt.Sprintf("tool-invocation-%d", time.Now().UnixNano()),
			Logger:       a.logger,
		}
	}
	toolResultMessages := make([]types.Message, 0)

	for _, toolCall := range toolCalls {
		if toolCall.Function.Name == "" {
			continue
		}

		startEvent := types.NewStreamingStatusMessage(
			fmt.Sprintf("tool-start-%s", toolCall.Id),
			types.TaskStateWorking,
			// "started", // started is not a valid task state according to spec
			map[string]any{
				"tool_name": toolCall.Function.Name,
			},
		)

		select {
		case outputChan <- types.NewMessageEvent("adk.agent.tool.started", startEvent.MessageID, startEvent, nil):
		case <-ctx.Done():
			return toolResultMessages
		}

		var args map[string]any
		var result string
		var toolErr error

		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			a.logger.Error("failed to parse tool arguments", zap.String("tool", toolCall.Function.Name), zap.Error(err))
			result = fmt.Sprintf("Error parsing tool arguments: %s", err.Error())
			toolErr = err

			failedEvent := types.NewStreamingStatusMessage(
				fmt.Sprintf("tool-failed-%s", toolCall.Id),
				types.TaskStateFailed,
				map[string]any{
					"tool_name": toolCall.Function.Name,
				},
			)

			select {
			case outputChan <- types.NewMessageEvent("adk.agent.tool.failed", failedEvent.MessageID, failedEvent, nil):
			case <-ctx.Done():
			}
		} else {
			// Execute BeforeTool callback if configured
			var toolResult map[string]interface{}
			if a.callbackExecutor != nil && callbackContext != nil {

				toolContext := &ToolContext{
					AgentName:    callbackContext.AgentName,
					InvocationID: callbackContext.InvocationID,
					Logger:       callbackContext.Logger,
				}

				tool, found := a.toolBox.GetTool(toolCall.Function.Name)
				if !found {
					a.logger.Error("failed to find tool", zap.String("tool", toolCall.Function.Name), zap.Error(toolErr))
					// FIXME: fix linting issue

					result = fmt.Sprintf("Tool execution failed: %s", toolErr.Error())
				}
				toolResult = a.callbackExecutor.ExecuteBeforeTool(ctx, tool, args, toolContext)
			}

			if toolResult != nil {
				// Callback returned a result, use it instead of executing tool
				resultBytes, err := json.Marshal(toolResult)
				if err != nil {
					result = fmt.Sprintf("Error marshaling callback result: %s", err.Error())
				} else {
					result = string(resultBytes)
				}
				a.logger.Debug("BeforeTool callback provided result, skipping tool execution", zap.String("tool", toolCall.Function.Name))
			} else if toolCall.Function.Name == "input_required" {
				a.logger.Debug("input_required tool called in streaming mode",
					zap.String("tool_call_id", toolCall.Id),
					zap.String("message", toolCall.Function.Arguments))

				result, toolErr = a.toolBox.ExecuteTool(ctx, toolCall.Function.Name, args)

				completedEvent := types.NewStreamingStatusMessage(
					fmt.Sprintf("tool-completed-%s", toolCall.Id),
					types.TaskStateCompleted,
					map[string]any{
						"tool_name": toolCall.Function.Name,
					},
				)

				select {
				case outputChan <- types.NewMessageEvent("adk.agent.tool.completed", completedEvent.MessageID, completedEvent, nil):
				case <-ctx.Done():
					return toolResultMessages
				}

				toolResultMessage := types.NewToolResultMessage(toolCall.Id, toolCall.Function.Name, result, toolErr != nil)

				select {
				case outputChan <- types.NewMessageEvent("adk.agent.tool.result", toolResultMessage.MessageID, toolResultMessage, nil):
				case <-ctx.Done():
					return toolResultMessages
				}

				toolResultMessages = append(toolResultMessages, *toolResultMessage)

				inputMessage := args["message"].(string)
				inputRequiredMessage := types.NewInputRequiredMessage(toolCall.Id, inputMessage)

				select {
				case outputChan <- types.NewMessageEvent("adk.agent.input.required", inputRequiredMessage.MessageID, inputRequiredMessage, nil):
				case <-ctx.Done():
				}

				toolResultMessages = append(toolResultMessages, *inputRequiredMessage)

				return toolResultMessages
			} else {
				// Normal tool execution
				result, toolErr = a.toolBox.ExecuteTool(ctx, toolCall.Function.Name, args)
			}

			// Execute AfterTool callback if configured and tool was executed (not from callback)
			if a.callbackExecutor != nil && callbackContext != nil && toolResult == nil {
				toolContext := &ToolContext{
					AgentName:    callbackContext.AgentName,
					InvocationID: callbackContext.InvocationID,
					Logger:       callbackContext.Logger,
				}

				// Convert result to map for callback
				originalResult := map[string]interface{}{"result": result}
				if toolErr != nil {
					originalResult["error"] = toolErr.Error()
				}

				tool, found := a.toolBox.GetTool(toolCall.Function.Name)
				if !found {
					a.logger.Error("failed to find tool", zap.String("tool", toolCall.Function.Name), zap.Error(toolErr))
					result = fmt.Sprintf("Tool execution failed: %s", toolErr.Error())
				}

				modifiedResult := a.callbackExecutor.ExecuteAfterTool(ctx, tool, args, toolContext, originalResult)
				if modifiedResult != nil {
					// Use modified result
					resultBytes, err := json.Marshal(modifiedResult)
					if err != nil {
						result = fmt.Sprintf("Error marshaling modified result: %s", err.Error())
					} else {
						result = string(resultBytes)
						// Clear toolErr if the callback fixed it
						if _, hasError := modifiedResult["error"]; !hasError {
							toolErr = nil
						}
					}
					a.logger.Debug("AfterTool callback modified result", zap.String("tool", toolCall.Function.Name))
				}
			}

			if toolErr != nil {
				a.logger.Error("failed to execute tool", zap.String("tool", toolCall.Function.Name), zap.Error(toolErr))
				result = fmt.Sprintf("Tool execution failed: %s", toolErr.Error())

				failedEvent := types.NewStreamingStatusMessage(
					fmt.Sprintf("tool-failed-%s", toolCall.Id),
					types.TaskStateFailed,
					map[string]any{
						"tool_name": toolCall.Function.Name,
					},
				)

				select {
				case outputChan <- types.NewMessageEvent("adk.agent.tool.failed", failedEvent.MessageID, failedEvent, nil):
				case <-ctx.Done():
				}
			} else {
				completedEvent := types.NewStreamingStatusMessage(
					fmt.Sprintf("tool-completed-%s", toolCall.Id),
					types.TaskStateCompleted,
					map[string]any{
						"tool_name": toolCall.Function.Name,
					},
				)

				select {
				case outputChan <- types.NewMessageEvent("adk.agent.tool.completed", completedEvent.MessageID, completedEvent, nil):
				case <-ctx.Done():
					return toolResultMessages
				}
			}
		}

		toolResultMessage := types.NewToolResultMessage(toolCall.Id, toolCall.Function.Name, result, toolErr != nil)

		select {
		case outputChan <- types.NewMessageEvent("adk.agent.tool.result", toolResultMessage.MessageID, toolResultMessage, nil):
		case <-ctx.Done():
			return toolResultMessages
		}

		toolResultMessages = append(toolResultMessages, *toolResultMessage)
	}

	return toolResultMessages
}

// executeTools handles tool execution, including callbacks and special-cases.
// It returns the result, an error (if any), and any extra messages to emit.
// it executes tools in sequence of the toolCall order
func (a *OpenAICompatibleAgentImpl) executeTools(
	ctx context.Context,
	toolCalls []sdk.ChatCompletionMessageToolCall,
	callbackContext *CallbackContext,
) ([]types.Message, error) {

	return pipeline.New(
		a.emitToolStartedStage(),
		a.executeToolStage(callbackContext, a.callbackExecutor),
		a.emitToolCompletedStage(),
	).Run(ctx, toolCalls)
}

func (a *OpenAICompatibleAgentImpl) emitToolStartedStage() pipeline.Stage {
	return func(ctx context.Context, call sdk.ChatCompletionMessageToolCall) ([]types.Message, error) {
		startEvent := types.NewStreamingStatusMessage(
			fmt.Sprintf("tool-start-%s", call.Id),
			"started",
			map[string]any{"tool_name": call.Function.Name},
		)

		if err := a.eventPublisher.Publish(ctx, types.NewMessageEvent("adk.agent.tool.started", startEvent.MessageID, startEvent, nil)); err != nil {
			return nil, err
		}

		return []types.Message{*startEvent}, nil
	}
}

// executeToolStage - executes the tool and any before/after tool hooks
func (a *OpenAICompatibleAgentImpl) executeToolStage(callbackContext *CallbackContext, callbackExecutor CallbackExecutor) pipeline.Stage {
	return func(ctx context.Context, call sdk.ChatCompletionMessageToolCall) ([]types.Message, error) {
		var messages []types.Message

		if call.Function.Name == "" {
			return nil, nil
		}

		var args map[string]any

		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			_, stageErr := a.emitToolFailedStage()(ctx, call)
			if stageErr != nil {
				return nil, fmt.Errorf("emitting failure state: %w, wrapped err: %w", stageErr, err)
			}

			return nil, err
		}

		tool, found := a.toolBox.GetTool(call.Function.Name)
		if !found {
			return nil, nil
		}

		toolContext := ToolContext{
			AgentName:    callbackContext.AgentName,
			InvocationID: callbackContext.InvocationID,
			TaskID:       callbackContext.TaskID,
			ContextID:    callbackContext.ContextID,
			State:        callbackContext.State,
			Logger:       callbackContext.Logger,
		}

		// 1. Execute before tool hooks
		res := callbackExecutor.ExecuteBeforeTool(ctx, tool, args, &toolContext)
		if res != nil {
			// Short-circuit the tool execution
			resultBytes, _ := json.Marshal(res)
			msg := types.NewToolResultMessage(call.Id, call.Function.Name, string(resultBytes), false)
			return []types.Message{*msg}, nil
		}

		// 2. Execute the tool
		result, toolErr := a.toolBox.ExecuteTool(ctx, call.Function.Name, args)

		originalResult := map[string]interface{}{"result": result}
		if toolErr != nil {
			originalResult["error"] = toolErr.Error()
		}

		// 3. Execute after tool hooks
		modifiedResult := a.callbackExecutor.ExecuteAfterTool(ctx, tool, args, &toolContext, originalResult)
		if modifiedResult != nil {
			// Use modified result
			resultBytes, err := json.Marshal(modifiedResult)
			if err != nil {
				result = fmt.Sprintf("Error marshaling modified result: %s", err.Error())
			} else {
				result = string(resultBytes)
				// Clear toolErr if the callback fixed it
				if _, hasError := modifiedResult["error"]; !hasError {
					toolErr = nil
				}
			}
			msg := types.NewToolResultMessage(call.Id, call.Function.Name, string(resultBytes), false)
			messages = append(messages, *msg)
			a.logger.Debug("AfterTool callback modified result", zap.String("tool", call.Function.Name))
		}

		if toolErr != nil {
			msgs, err := a.emitToolFailedStage()(ctx, call)
			if err != nil {
				return nil, err
			}
			messages = append(messages, msgs...)
		} else {

			resultMsg := types.NewToolResultMessage(call.Id, call.Function.Name, result, toolErr != nil)
			if err := a.eventPublisher.Publish(ctx, types.NewMessageEvent("adk.agent.tool.result", resultMsg.MessageID, resultMsg, nil)); err != nil {
				return nil, err
			}

			messages = append(messages, *resultMsg)

			msgs, err := a.emitToolCompletedStage()(ctx, call)
			if err != nil {
				return nil, err
			}
			messages = append(messages, msgs...)
		}

		return messages, nil

	}
}

// emitToolCompletedStage sends a "adk.agent.tool.completed" event.
func (a *OpenAICompatibleAgentImpl) emitToolCompletedStage() pipeline.Stage {
	return func(ctx context.Context, call sdk.ChatCompletionMessageToolCall) ([]types.Message, error) {
		completedEvent := types.NewStreamingStatusMessage(
			fmt.Sprintf("tool-completed-%s", call.Id),
			types.TaskStateCompleted,
			map[string]any{
				"tool_name": call.Function.Name,
			},
		)

		if err := a.eventPublisher.Publish(ctx, types.NewMessageEvent("adk.agent.tool.completed", completedEvent.MessageID, completedEvent, nil)); err != nil {
			return nil, err
		}

		return []types.Message{*completedEvent}, nil
	}

}

// emitToolFailedStage sends a "adk.agent.tool.failed" event.
func (a *OpenAICompatibleAgentImpl) emitToolFailedStage() pipeline.Stage {
	return func(ctx context.Context, call sdk.ChatCompletionMessageToolCall) ([]types.Message, error) {
		if call.Function.Name == "" {
			return nil, nil
		}

		failedEvent := types.NewStreamingStatusMessage(
			"tool-failed-"+call.Id,
			"failed",
			map[string]any{"tool_name": call.Function.Name},
		)

		if err := a.eventPublisher.Publish(ctx, types.NewMessageEvent("adk.agent.tool.failed", failedEvent.MessageID, failedEvent, nil)); err != nil {
			return nil, err
		}

		return []types.Message{*failedEvent}, nil
	}
}

// isCompleteJSON checks if a string contains complete JSON by counting balanced braces
func isCompleteJSON(s string) bool {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "{") || !strings.HasSuffix(s, "}") {
		return false
	}

	openCount := 0
	for _, char := range s {
		switch char {
		case '{':
			openCount++
		case '}':
			openCount--
		}
	}

	return openCount == 0
}
