package server

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	cloudevents "github.com/cloudevents/sdk-go/v2"
	config "github.com/inference-gateway/adk/server/config"
	utils "github.com/inference-gateway/adk/server/utils"
	types "github.com/inference-gateway/adk/types"
	sdk "github.com/inference-gateway/sdk"
	zap "go.uber.org/zap"
)

// AgentResponse contains the response and any additional messages generated during agent execution
type AgentResponse struct {
	// Response is the main assistant response message
	Response *types.Message
	// AdditionalMessages contains any tool calls, tool responses, or intermediate messages
	// that should be added to the conversation history
	AdditionalMessages []types.Message
}

//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 . OpenAICompatibleAgent

// OpenAICompatibleAgent represents an agent that can interact with OpenAI-compatible LLM APIs and execute tools
// The agent is stateless and does not maintain conversation history
// Tools are configured during agent creation via the toolbox
type OpenAICompatibleAgent interface {
	// Run processes a conversation and returns the assistant's response along with any additional messages
	// Uses the agent's configured toolbox for tool execution
	Run(ctx context.Context, messages []types.Message) (*AgentResponse, error)

	// RunWithStream processes a conversation and returns a streaming response
	// Uses the agent's configured toolbox for tool execution
	RunWithStream(ctx context.Context, messages []types.Message) (<-chan cloudevents.Event, error)
}

// OpenAICompatibleAgentImpl is the implementation of OpenAICompatibleAgent
// This implementation is stateless and does not maintain conversation history
type OpenAICompatibleAgentImpl struct {
	logger           *zap.Logger
	llmClient        LLMClient
	toolBox          ToolBox
	converter        utils.MessageConverter
	config           *config.AgentConfig
	callbackExecutor CallbackExecutor
	// TODO: implement a default channel event publisher and include options for configuring own
	eventPublisher EventPublisher
}

// NewOpenAICompatibleAgent creates a new OpenAICompatibleAgentImpl
func NewOpenAICompatibleAgent(logger *zap.Logger) *OpenAICompatibleAgentImpl {
	defaultConfig := &config.AgentConfig{
		MaxChatCompletionIterations: 10,
		SystemPrompt:                "You are a helpful AI assistant.",
	}
	return &OpenAICompatibleAgentImpl{
		logger:    logger,
		converter: utils.NewOptimizedMessageConverter(logger),
		config:    defaultConfig,
	}
}

// NewOpenAICompatibleAgentWithConfig creates a new OpenAICompatibleAgentImpl with configuration
func NewOpenAICompatibleAgentWithConfig(logger *zap.Logger, cfg *config.AgentConfig) *OpenAICompatibleAgentImpl {
	return &OpenAICompatibleAgentImpl{
		logger:    logger,
		converter: utils.NewOptimizedMessageConverter(logger),
		config:    cfg,
	}
}

// NewOpenAICompatibleAgentWithLLM creates a new agent with an LLM client
func NewOpenAICompatibleAgentWithLLM(logger *zap.Logger, llmClient LLMClient) *OpenAICompatibleAgentImpl {
	agent := NewOpenAICompatibleAgent(logger)
	agent.llmClient = llmClient
	return agent
}

// NewOpenAICompatibleAgentWithLLMConfig creates a new agent with LLM configuration
func NewOpenAICompatibleAgentWithLLMConfig(logger *zap.Logger, config *config.AgentConfig) (*OpenAICompatibleAgentImpl, error) {
	client, err := NewOpenAICompatibleLLMClient(config, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create llm client: %w", err)
	}

	agent := NewOpenAICompatibleAgentWithConfig(logger, config)
	agent.llmClient = client
	return agent, nil
}

// SetLLMClient sets the LLM client for the agent
func (a *OpenAICompatibleAgentImpl) SetLLMClient(client LLMClient) {
	a.llmClient = client
}

// SetToolBox sets the tool box for the agent
func (a *OpenAICompatibleAgentImpl) SetToolBox(toolBox ToolBox) {
	a.toolBox = toolBox
}

// SetCallbackExecutor sets the callback executor for the agent
func (a *OpenAICompatibleAgentImpl) SetCallbackExecutor(executor CallbackExecutor) {
	a.callbackExecutor = executor
}

// GetCallbackExecutor returns the callback executor for the agent if available or a provided default
func (a *OpenAICompatibleAgentImpl) GetCallbackExecutor() CallbackExecutor {
	if a.callbackExecutor == nil {
		return NewCallbackExecutor(nil, a.logger)
	}
	return a.callbackExecutor
}

// Run processes a conversation and returns the assistant's response along with additional messages
// TODO: add more tests which use the callbacks
func (a *OpenAICompatibleAgentImpl) Run(ctx context.Context, messages []types.Message) (*AgentResponse, error) {
	if a.llmClient == nil {
		return nil, fmt.Errorf("no LLM client configured for agent")
	}

	callbackContext := &CallbackContext{
		AgentName:    a.config.Model, // TODO: somehow we need to get the agent name
		InvocationID: fmt.Sprintf("invocation-%d", time.Now().UnixNano()),
		Logger:       a.logger,
	}

	skipResult := a.GetCallbackExecutor().ExecuteBeforeAgent(ctx, callbackContext)
	if skipResult != nil {
		// Callback returned a message, skip agent execution
		return &AgentResponse{
			Response:           skipResult,
			AdditionalMessages: []types.Message{},
		}, nil
	}

	conversation, err := a.converter.ConvertToSDK(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages to SDK format: %w", err)
	}

	if a.config != nil && a.config.SystemPrompt != "" {
		systemMessage := sdk.Message{
			Role:    sdk.System,
			Content: a.config.SystemPrompt,
		}
		conversation = append([]sdk.Message{systemMessage}, conversation...)
	}

	maxIterations := 10
	if a.config != nil && a.config.MaxChatCompletionIterations > 0 {
		maxIterations = a.config.MaxChatCompletionIterations
	}

	var tools []sdk.ChatCompletionTool
	if a.toolBox != nil {
		tools = a.toolBox.GetTools()
	}

	var additionalMessages []types.Message

	callbackContext = &CallbackContext{
		AgentName:    "agent", // TODO: Get actual agent name from context
		InvocationID: fmt.Sprintf("invocation-%d", time.Now().UnixNano()),
		Logger:       a.logger,
		// TODO: fill out other fields
	}

	for iteration := 0; iteration < maxIterations; iteration++ {
		// Execute BeforeModel callback if configured
		var llmResponse *LLMResponse
		// Convert conversation to LLM request format for callback
		a2aMessages := make([]types.Message, len(conversation))
		for i, msg := range conversation {
			a2aMsg, err := a.converter.ConvertFromSDK(msg)
			if err != nil {
				a.logger.Warn("failed to convert SDK message for callback", zap.Error(err))
				continue
			}
			a2aMessages[i] = *a2aMsg
		}

		llmRequest := &LLMRequest{
			Contents: a2aMessages,
			Config:   &LLMConfig{
				// TODO: Add system instruction from config
			},
		}

		llmResponse = a.GetCallbackExecutor().ExecuteBeforeModel(ctx, callbackContext, llmRequest)

		var assistantMessage sdk.Message
		if llmResponse != nil {
			// Callback returned a response, use it instead of calling LLM
			sdkMsg, err := a.converter.ConvertToSDK([]types.Message{*llmResponse.Content})
			if err != nil {
				return nil, fmt.Errorf("failed to convert callback response to SDK format: %w", err)
			}
			if len(sdkMsg) > 0 {
				assistantMessage = sdkMsg[0]
			}
		} else {
			// Normal LLM call
			response, err := a.llmClient.CreateChatCompletion(ctx, conversation, tools...)
			if err != nil {
				return nil, fmt.Errorf("failed to create chat completion: %w", err)
			}

			if len(response.Choices) == 0 {
				return nil, fmt.Errorf("no choices returned from LLM")
			}

			assistantMessage = response.Choices[0].Message

			// Execute AfterModel callback if configured
			// Convert to LLMResponse for callback
			a2aMsg, err := a.converter.ConvertFromSDK(assistantMessage)
			if err != nil {
				a.logger.Warn("failed to convert assistant message for callback", zap.Error(err))
			} else {
				originalResponse := &LLMResponse{
					Content: a2aMsg,
				}

				modifiedResponse := a.GetCallbackExecutor().ExecuteAfterModel(ctx, callbackContext, originalResponse)
				if modifiedResponse != nil {
					// Use modified response
					sdkMsg, err := a.converter.ConvertToSDK([]types.Message{*modifiedResponse.Content})
					if err == nil && len(sdkMsg) > 0 {
						assistantMessage = sdkMsg[0]
					}
				}
			}
		}

		conversation = append(conversation, assistantMessage)

		assistantA2A, err := a.converter.ConvertFromSDK(assistantMessage)
		if err != nil {
			return nil, fmt.Errorf("failed to convert assistant message to A2A format: %w", err)
		}

		if assistantMessage.ToolCalls == nil || len(*assistantMessage.ToolCalls) == 0 || a.toolBox == nil {
			finalResult := &AgentResponse{
				Response:           assistantA2A,
				AdditionalMessages: additionalMessages,
			}

			// Execute AfterAgent callback if configured
			modifiedResponse := a.GetCallbackExecutor().ExecuteAfterAgent(ctx, callbackContext, finalResult.Response)
			if modifiedResponse != nil {
				finalResult.Response = modifiedResponse
			}

			return finalResult, nil
		}

		additionalMessages = append(additionalMessages, *assistantA2A)

		for _, toolCall := range *assistantMessage.ToolCalls {
			if toolCall.Function.Name == "" {
				continue
			}

			var args map[string]any
			var result string
			var toolErr error

			err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
			if err != nil {
				a.logger.Error("failed to parse tool arguments", zap.String("tool", toolCall.Function.Name), zap.Error(err))
				return &AgentResponse{
					Response: &types.Message{
						Kind:      "message",
						MessageID: fmt.Sprintf("tool-error-%s", toolCall.Id),
						Role:      "tool",
						Parts: []types.Part{
							map[string]any{
								"kind": "text",
								"text": fmt.Sprintf("Error parsing tool arguments: %s", err.Error()),
							},
						},
					},
				}, err
			}

			if toolCall.Function.Name == "input_required" {
				a.logger.Debug("input_required tool called",
					zap.String("tool_call_id", toolCall.Id),
					zap.String("message", toolCall.Function.Arguments))
				inputMessage := args["message"].(string)
				return &AgentResponse{
					Response: &types.Message{
						Kind:      "input_required",
						MessageID: fmt.Sprintf("input-required-%s", toolCall.Id),
						Role:      "assistant",
						Parts: []types.Part{
							map[string]any{
								"kind": "text",
								"text": inputMessage,
							},
						},
					},
				}, nil
			}

			// Execute BeforeTool callback if configured
			var toolResult map[string]interface{}

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

			toolResult = a.GetCallbackExecutor().ExecuteBeforeTool(ctx, tool, args, toolContext)

			if toolResult != nil {
				// Callback returned a result, use it instead of executing tool
				resultBytes, err := json.Marshal(toolResult)
				if err != nil {
					result = fmt.Sprintf("Error marshaling callback result: %s", err.Error())
				} else {
					result = string(resultBytes)
				}
			} else {
				// Normal tool execution
				result, toolErr = a.toolBox.ExecuteTool(ctx, toolCall.Function.Name, args)
				if toolErr != nil {
					a.logger.Error("failed to execute tool", zap.String("tool", toolCall.Function.Name), zap.Error(toolErr))
					result = fmt.Sprintf("Tool execution failed: %s", toolErr.Error())
				}

				// Execute AfterTool callback if configured

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

				modifiedResult := a.GetCallbackExecutor().ExecuteAfterTool(ctx, tool, args, toolContext, originalResult)
				if modifiedResult != nil {
					// Use modified result
					resultBytes, err := json.Marshal(modifiedResult)
					if err != nil {
						result = fmt.Sprintf("Error marshaling modified result: %s", err.Error())
					} else {
						result = string(resultBytes)
					}
				}
			}

			toolMessage := sdk.Message{
				Role:       sdk.Tool,
				Content:    result,
				ToolCallId: &toolCall.Id,
			}
			conversation = append(conversation, toolMessage)

			toolA2A := &types.Message{
				Kind:      "message",
				MessageID: fmt.Sprintf("tool-%s-%d", toolCall.Function.Name, time.Now().UnixNano()),
				Role:      "tool",
				Parts: []types.Part{
					map[string]any{
						"kind": "data",
						"data": map[string]any{
							"tool_call_id": toolCall.Id,
							"tool_name":    toolCall.Function.Name,
							"result":       result,
							"error":        toolErr != nil,
						},
					},
				},
			}
			additionalMessages = append(additionalMessages, *toolA2A)
		}
	}

	finalResult := &AgentResponse{
		Response: &types.Message{
			Kind:      "message",
			MessageID: fmt.Sprintf("error-%d", time.Now().UnixNano()),
			Role:      "assistant",
			Parts: []types.Part{
				map[string]any{
					"kind": "text",
					"text": fmt.Sprintf("Maximum iterations (%d) reached without final response", maxIterations),
				},
			},
		},
		AdditionalMessages: additionalMessages,
	}

	// Execute AfterAgent callback if configured
	modifiedResponse := a.GetCallbackExecutor().ExecuteAfterAgent(ctx, callbackContext, finalResult.Response)
	if modifiedResponse != nil {
		finalResult.Response = modifiedResponse
	}

	return finalResult, fmt.Errorf("maximum iterations (%d) reached without final response", maxIterations)
}
