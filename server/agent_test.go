package server_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/inference-gateway/adk/server"
	"github.com/inference-gateway/adk/server/config"
	"github.com/inference-gateway/adk/server/mocks"
	"github.com/inference-gateway/adk/types"
	"github.com/inference-gateway/sdk"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestOpenAICompatibleAgentImpl_Run_NoLLMClient tests the Run method when no LLM client is configured
func TestOpenAICompatibleAgentImpl_Run_NoLLMClient(t *testing.T) {
	logger := zap.NewNop()
	agent := server.NewOpenAICompatibleAgent(logger)

	messages := []types.Message{
		{
			Role: "user",
			Parts: []types.Part{
				map[string]any{
					"kind": "text",
					"text": "Hello",
				},
			},
		},
	}

	result, err := agent.Run(context.Background(), messages)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no LLM client configured for agent")
	assert.Nil(t, result)
}

// TestOpenAICompatibleAgentImpl_Run_BasicConversation tests basic conversation flow
func TestOpenAICompatibleAgentImpl_Run_BasicConversation(t *testing.T) {
	tests := []struct {
		name             string
		inputMessages    []types.Message
		llmResponse      *sdk.CreateChatCompletionResponse
		llmError         error
		expectedResponse *types.Message
		expectedError    string
		systemPrompt     string
		maxIterations    int
	}{
		{
			name: "successful_simple_conversation",
			inputMessages: []types.Message{
				{
					Role: "user",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Hello",
						},
					},
				},
			},
			llmResponse: &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{
					{
						Message: sdk.Message{
							Role:    sdk.Assistant,
							Content: "Hello! How can I help you today?",
						},
					},
				},
			},
			expectedResponse: &types.Message{
				Role: "assistant",
				Parts: []types.Part{
					map[string]any{
						"kind": "text",
						"text": "Hello! How can I help you today?",
					},
				},
			},
		},
		{
			name: "llm_client_error",
			inputMessages: []types.Message{
				{
					Role: "user",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Hello",
						},
					},
				},
			},
			llmError:      errors.New("LLM service unavailable"),
			expectedError: "failed to create chat completion: LLM service unavailable",
		},
		{
			name: "no_choices_returned",
			inputMessages: []types.Message{
				{
					Role: "user",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Hello",
						},
					},
				},
			},
			llmResponse: &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{},
			},
			expectedError: "no choices returned from LLM",
		},
		{
			name: "with_system_prompt",
			inputMessages: []types.Message{
				{
					Role: "user",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Hello",
						},
					},
				},
			},
			llmResponse: &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{
					{
						Message: sdk.Message{
							Role:    sdk.Assistant,
							Content: "Greetings! I'm your specialized assistant.",
						},
					},
				},
			},
			systemPrompt: "You are a specialized AI assistant for testing purposes.",
			expectedResponse: &types.Message{
				Role: "assistant",
				Parts: []types.Part{
					map[string]any{
						"kind": "text",
						"text": "Greetings! I'm your specialized assistant.",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := zap.NewNop()
			mockLLMClient := &mocks.FakeLLMClient{}

			if tt.llmError != nil {
				mockLLMClient.CreateChatCompletionReturns(nil, tt.llmError)
			} else if tt.llmResponse != nil {
				mockLLMClient.CreateChatCompletionReturns(tt.llmResponse, nil)
			}

			cfg := &config.AgentConfig{
				SystemPrompt:                tt.systemPrompt,
				MaxChatCompletionIterations: tt.maxIterations,
			}
			if tt.maxIterations == 0 {
				cfg.MaxChatCompletionIterations = 10 // default
			}

			agent := server.NewOpenAICompatibleAgentWithConfig(logger, cfg)
			agent.SetLLMClient(mockLLMClient)

			result, err := agent.Run(context.Background(), tt.inputMessages)

			if tt.expectedError != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
				assert.Nil(t, result)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, result)
				assert.NotNil(t, result.Response)
				assert.Equal(t, tt.expectedResponse.Role, result.Response.Role)
				// Verify the response content (simplified comparison)
				if len(tt.expectedResponse.Parts) > 0 && len(result.Response.Parts) > 0 {
					expectedText := tt.expectedResponse.Parts[0].(map[string]any)["text"]
					actualText := result.Response.Parts[0].(map[string]any)["text"]
					assert.Equal(t, expectedText, actualText)
				}
			}

			// Verify LLM client was called appropriately
			if tt.llmError != nil || tt.llmResponse != nil {
				assert.Equal(t, 1, mockLLMClient.CreateChatCompletionCallCount())

				// Verify system prompt was added if configured
				ctx, messages, tools := mockLLMClient.CreateChatCompletionArgsForCall(0)
				assert.NotNil(t, ctx)
				assert.NotNil(t, messages)
				if tt.systemPrompt != "" {
					assert.True(t, len(messages) >= 2) // system + user message
					assert.Equal(t, sdk.System, messages[0].Role)
					assert.Equal(t, tt.systemPrompt, messages[0].Content)
				}
				assert.Empty(t, tools) // no tools in basic tests
			}
		})
	}
}

// TestOpenAICompatibleAgentImpl_Run_MaxIterations tests maximum iterations behavior
func TestOpenAICompatibleAgentImpl_Run_MaxIterations(t *testing.T) {
	logger := zap.NewNop()
	mockLLMClient := &mocks.FakeLLMClient{}

	// Create a tool response that will trigger multiple iterations
	toolCalls := []sdk.ChatCompletionMessageToolCall{
		{
			Id: "call_1",
			Function: sdk.ChatCompletionMessageToolCallFunction{
				Name:      "test_tool",
				Arguments: `{"param": "value"}`,
			},
		},
	}

	// First response: assistant message with tool call
	firstResponse := &sdk.CreateChatCompletionResponse{
		Choices: []sdk.ChatCompletionChoice{
			{
				Message: sdk.Message{
					Role:      sdk.Assistant,
					Content:   "I'll help you with that.",
					ToolCalls: &toolCalls,
				},
			},
		},
	}

	// Subsequent responses: more tool calls to exceed max iterations
	mockLLMClient.CreateChatCompletionReturnsOnCall(0, firstResponse, nil)
	for i := 1; i < 10; i++ {
		mockLLMClient.CreateChatCompletionReturnsOnCall(i, firstResponse, nil)
	}

	cfg := &config.AgentConfig{
		MaxChatCompletionIterations: 2, // Set low to test max iterations
	}
	agent := server.NewOpenAICompatibleAgentWithConfig(logger, cfg)
	agent.SetLLMClient(mockLLMClient)

	// Set up a simple toolbox with a test tool
	toolBox := server.NewDefaultToolBox()
	testTool := server.NewBasicTool(
		"test_tool",
		"A test tool for unit testing",
		map[string]any{"type": "object"},
		func(ctx context.Context, args map[string]any) (string, error) {
			return "Tool executed successfully", nil
		},
	)
	toolBox.AddTool(testTool)
	agent.SetToolBox(toolBox)

	messages := []types.Message{
		{
			Role: "user",
			Parts: []types.Part{
				map[string]any{
					"kind": "text",
					"text": "Please use the test tool repeatedly",
				},
			},
		},
	}

	result, err := agent.Run(context.Background(), messages)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "maximum iterations (2) reached without final response")
	assert.NotNil(t, result)
	assert.NotNil(t, result.Response)
	assert.Equal(t, "assistant", result.Response.Role)

	// Verify the error message in the response
	if len(result.Response.Parts) > 0 {
		part := result.Response.Parts[0].(map[string]any)
		assert.Equal(t, "text", part["kind"])
		assert.Contains(t, part["text"], "Maximum iterations (2) reached without final response")
	}

	// Should have made exactly maxIterations LLM calls
	assert.Equal(t, 2, mockLLMClient.CreateChatCompletionCallCount())
}

// TestOpenAICompatibleAgentImpl_Run_ToolExecution tests tool execution scenarios
func TestOpenAICompatibleAgentImpl_Run_ToolExecution(t *testing.T) {
	tests := []struct {
		name             string
		toolName         string
		toolArgs         string
		toolResult       string
		toolError        error
		expectedInResult string
		expectError      bool
		setupTool        bool
	}{
		{
			name:             "successful_tool_execution",
			toolName:         "test_tool",
			toolArgs:         `{"param": "test_value"}`,
			toolResult:       "Tool executed successfully",
			expectedInResult: "Tool executed successfully",
			setupTool:        true,
		},
		{
			name:             "tool_execution_error",
			toolName:         "failing_tool",
			toolArgs:         `{"param": "test_value"}`,
			toolError:        errors.New("tool failed"),
			expectedInResult: "Tool execution failed: tool failed",
			setupTool:        true,
		},
		{
			name:             "tool_not_found",
			toolName:         "nonexistent_tool",
			toolArgs:         `{"param": "test_value"}`,
			expectedInResult: "tool not found",
			setupTool:        false,
		},
		{
			name:        "invalid_tool_arguments",
			toolName:    "test_tool",
			toolArgs:    `{invalid json}`,
			expectError: true,
			setupTool:   true,
		},
		{
			name:             "input_required_tool",
			toolName:         "input_required",
			toolArgs:         `{"message": "Please provide more information"}`,
			expectedInResult: "Please provide more information",
			setupTool:        true, // input_required is handled specially
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := zap.NewNop()
			mockLLMClient := &mocks.FakeLLMClient{}

			// Create tool calls for the test
			toolCalls := []sdk.ChatCompletionMessageToolCall{
				{
					Id: "call_1",
					Function: sdk.ChatCompletionMessageToolCallFunction{
						Name:      tt.toolName,
						Arguments: tt.toolArgs,
					},
				},
			}

			// First response with tool call
			firstResponse := &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{
					{
						Message: sdk.Message{
							Role:      sdk.Assistant,
							Content:   "I'll help you with that.",
							ToolCalls: &toolCalls,
						},
					},
				},
			}

			// Second response (final response after tool execution)
			secondResponse := &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{
					{
						Message: sdk.Message{
							Role:    sdk.Assistant,
							Content: "Task completed successfully.",
						},
					},
				},
			}

			mockLLMClient.CreateChatCompletionReturnsOnCall(0, firstResponse, nil)
			// if !tt.expectError && (tt.toolName != "input_required" && tt.toolName != "nonexistent_tool") {
			if !tt.expectError && (tt.toolName != "input_required") {
				mockLLMClient.CreateChatCompletionReturnsOnCall(1, secondResponse, nil)
			}

			agent := server.NewOpenAICompatibleAgent(logger)
			agent.SetLLMClient(mockLLMClient)

			// Set up toolbox - always set up a toolbox so tool processing occurs
			// toolBox := server.NewDefaultToolBox()
			toolBox := server.NewToolBox()
			if tt.setupTool {
				testTool := server.NewBasicTool(
					tt.toolName,
					"Test tool for unit testing",
					map[string]any{"type": "object"},
					func(ctx context.Context, args map[string]any) (string, error) {
						if tt.toolError != nil {
							return "", tt.toolError
						}
						return tt.toolResult, nil
					},
				)
				toolBox.AddTool(testTool)
			}
			agent.SetToolBox(toolBox)

			messages := []types.Message{
				{
					Role: "user",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Please use the tool",
						},
					},
				},
			}

			result, err := agent.Run(context.Background(), messages)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.NotNil(t, result)
			assert.NotNil(t, result.Response)

			// Debug: print the actual response for failing tests
			if tt.toolName == "nonexistent_tool" || tt.toolName == "input_required" {
				t.Logf("Response for %s: %+v", tt.name, result.Response)
				t.Logf("AdditionalMessages count: %d", len(result.AdditionalMessages))
				for i, msg := range result.AdditionalMessages {
					t.Logf("AdditionalMessage %d: Role=%s, Parts=%+v", i, msg.Role, msg.Parts)
				}
			}

			// For input_required, check the specific response format
			if tt.toolName == "input_required" {
				assert.Equal(t, "input_required", result.Response.Kind)
				assert.Equal(t, "assistant", result.Response.Role)
				if len(result.Response.Parts) > 0 {
					part := result.Response.Parts[0].(map[string]any)
					assert.Contains(t, part["text"], tt.expectedInResult)
				}
			} else {
				// For other tools, should have additional messages from tool execution
				if tt.expectedInResult != "" && tt.toolName != "nonexistent_tool" {
					assert.True(t, len(result.AdditionalMessages) > 0)
					// Find tool message in additional messages
					foundToolMessage := false
					for _, msg := range result.AdditionalMessages {
						if msg.Role == "tool" && len(msg.Parts) > 0 {
							if data, ok := msg.Parts[0].(map[string]any); ok {
								if dataMap, ok := data["data"].(map[string]any); ok {
									if result, ok := dataMap["result"].(string); ok {

										fmt.Println("result: ", result)

										foundToolMessage = true
										assert.Contains(t, result, tt.expectedInResult)
									}

								}
							}
						}
					}
					// For tool_not_found case, we expect the tool execution to fail but continue
					if tt.toolName == "nonexistent_tool" && len(result.AdditionalMessages) > 0 {
						// Check if we have a tool message with error
						for _, msg := range result.AdditionalMessages {
							if msg.Role == "tool" {
								foundToolMessage = true
								break
							}
						}
					}
					if tt.setupTool || tt.toolError != nil {
						assert.True(t, foundToolMessage, "Expected to find tool result in additional messages for test: %s", tt.name)
					}
				}
			}

			// Verify LLM calls
			if tt.toolName == "input_required" {
				assert.Equal(t, 1, mockLLMClient.CreateChatCompletionCallCount())
			} else if !tt.expectError {
				// For tool_not_found, only one LLM call is made since tool execution fails early
				expectedCalls := 2 // First call for tool, second call after tool execution
				if tt.toolName == "nonexistent_tool" {
					expectedCalls = 1 // Only one call since tool is not found and execution stops
				}
				assert.Equal(t, expectedCalls, mockLLMClient.CreateChatCompletionCallCount())
			}
		})
	}
}

// TestOpenAICompatibleAgentImpl_Run_CallbackIntegration tests callback integration
func TestOpenAICompatibleAgentImpl_Run_CallbackIntegration(t *testing.T) {
	tests := []struct {
		name                     string
		beforeAgentCallback      server.BeforeAgentCallback
		afterAgentCallback       server.AfterAgentCallback
		beforeModelCallback      server.BeforeModelCallback
		afterModelCallback       server.AfterModelCallback
		expectedSkipLLM          bool
		expectedResponse         string
		expectedCallbackExecuted bool
	}{
		{
			name: "before_agent_callback_skips_execution",
			beforeAgentCallback: func(ctx context.Context, callbackContext *server.CallbackContext) *types.Message {
				return &types.Message{
					Role: "assistant",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Skipped by before agent callback",
						},
					},
				}
			},
			expectedSkipLLM:  true,
			expectedResponse: "Skipped by before agent callback",
		},
		{
			name: "after_agent_callback_modifies_response",
			afterAgentCallback: func(ctx context.Context, callbackContext *server.CallbackContext, agentOutput *types.Message) *types.Message {
				return &types.Message{
					Role: "assistant",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Modified by after agent callback",
						},
					},
				}
			},
			expectedSkipLLM:  false,
			expectedResponse: "Modified by after agent callback",
		},
		{
			name: "before_model_callback_skips_llm",
			beforeModelCallback: func(ctx context.Context, callbackContext *server.CallbackContext, llmRequest *server.LLMRequest) *server.LLMResponse {
				return &server.LLMResponse{
					Content: &types.Message{
						Role: "assistant",
						Parts: []types.Part{
							map[string]any{
								"kind": "text",
								"text": "Response from before model callback",
							},
						},
					},
				}
			},
			expectedSkipLLM:  true,
			expectedResponse: "Response from before model callback",
		},
		{
			name: "after_model_callback_modifies_llm_response",
			afterModelCallback: func(ctx context.Context, callbackContext *server.CallbackContext, llmResponse *server.LLMResponse) *server.LLMResponse {
				return &server.LLMResponse{
					Content: &types.Message{
						Role: "assistant",
						Parts: []types.Part{
							map[string]any{
								"kind": "text",
								"text": "Modified by after model callback",
							},
						},
					},
				}
			},
			expectedSkipLLM:  false,
			expectedResponse: "Modified by after model callback",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := zap.NewNop()
			mockLLMClient := &mocks.FakeLLMClient{}

			// Set up normal LLM response for cases that don't skip LLM
			normalResponse := &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{
					{
						Message: sdk.Message{
							Role:    sdk.Assistant,
							Content: "Normal LLM response",
						},
					},
				},
			}
			mockLLMClient.CreateChatCompletionReturns(normalResponse, nil)

			// Create callback configuration
			callbackConfig := &server.CallbackConfig{}
			if tt.beforeAgentCallback != nil {
				callbackConfig.BeforeAgent = []server.BeforeAgentCallback{tt.beforeAgentCallback}
			}
			if tt.afterAgentCallback != nil {
				callbackConfig.AfterAgent = []server.AfterAgentCallback{tt.afterAgentCallback}
			}
			if tt.beforeModelCallback != nil {
				callbackConfig.BeforeModel = []server.BeforeModelCallback{tt.beforeModelCallback}
			}
			if tt.afterModelCallback != nil {
				callbackConfig.AfterModel = []server.AfterModelCallback{tt.afterModelCallback}
			}

			callbackExecutor := server.NewCallbackExecutor(callbackConfig, logger)

			agent := server.NewOpenAICompatibleAgent(logger)
			agent.SetLLMClient(mockLLMClient)
			agent.SetCallbackExecutor(callbackExecutor)

			messages := []types.Message{
				{
					Role: "user",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Hello",
						},
					},
				},
			}

			result, err := agent.Run(context.Background(), messages)

			require.NoError(t, err)
			assert.NotNil(t, result)
			assert.NotNil(t, result.Response)

			// Verify response content
			if len(result.Response.Parts) > 0 {
				part := result.Response.Parts[0].(map[string]any)
				actualText := part["text"].(string)
				assert.Equal(t, tt.expectedResponse, actualText)
			}

			// Verify LLM call expectations
			if tt.expectedSkipLLM {
				assert.Equal(t, 0, mockLLMClient.CreateChatCompletionCallCount(), "LLM should have been skipped")
			} else {
				assert.Equal(t, 1, mockLLMClient.CreateChatCompletionCallCount(), "LLM should have been called")
			}
		})
	}
}

// TestOpenAICompatibleAgentImpl_Run_ToolCallbackIntegration tests tool-related callback integration
func TestOpenAICompatibleAgentImpl_Run_ToolCallbackIntegration(t *testing.T) {
	tests := []struct {
		name               string
		beforeToolCallback server.BeforeToolCallback
		afterToolCallback  server.AfterToolCallback
		expectedToolResult string
		expectedSkipTool   bool
	}{
		{
			name: "before_tool_callback_skips_tool_execution",
			beforeToolCallback: func(ctx context.Context, tool server.Tool, args map[string]interface{}, toolContext *server.ToolContext) map[string]interface{} {
				return map[string]interface{}{
					"result": "Skipped tool execution via callback",
				}
			},
			expectedToolResult: "Skipped tool execution via callback",
			expectedSkipTool:   true,
		},
		{
			name: "after_tool_callback_modifies_result",
			afterToolCallback: func(ctx context.Context, tool server.Tool, args map[string]interface{}, toolContext *server.ToolContext, toolResult map[string]interface{}) map[string]interface{} {
				return map[string]interface{}{
					"result": "Modified by after tool callback",
				}
			},
			expectedToolResult: "Modified by after tool callback",
			expectedSkipTool:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := zap.NewNop()
			mockLLMClient := &mocks.FakeLLMClient{}

			// Tool call setup
			toolCalls := []sdk.ChatCompletionMessageToolCall{
				{
					Id: "call_1",
					Function: sdk.ChatCompletionMessageToolCallFunction{
						Name:      "test_tool",
						Arguments: `{"param": "value"}`,
					},
				},
			}

			firstResponse := &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{
					{
						Message: sdk.Message{
							Role:      sdk.Assistant,
							Content:   "Using tool.",
							ToolCalls: &toolCalls,
						},
					},
				},
			}

			secondResponse := &sdk.CreateChatCompletionResponse{
				Choices: []sdk.ChatCompletionChoice{
					{
						Message: sdk.Message{
							Role:    sdk.Assistant,
							Content: "Tool completed.",
						},
					},
				},
			}

			mockLLMClient.CreateChatCompletionReturnsOnCall(0, firstResponse, nil)
			mockLLMClient.CreateChatCompletionReturnsOnCall(1, secondResponse, nil)

			// Set up toolbox
			toolBox := server.NewDefaultToolBox()
			toolExecuted := false
			testTool := server.NewBasicTool(
				"test_tool",
				"Test tool for callback testing",
				map[string]any{"type": "object"},
				func(ctx context.Context, args map[string]any) (string, error) {
					toolExecuted = true
					return "Original tool result", nil
				},
			)
			toolBox.AddTool(testTool)

			// Create callback configuration
			callbackConfig := &server.CallbackConfig{}
			if tt.beforeToolCallback != nil {
				callbackConfig.BeforeTool = []server.BeforeToolCallback{tt.beforeToolCallback}
			}
			if tt.afterToolCallback != nil {
				callbackConfig.AfterTool = []server.AfterToolCallback{tt.afterToolCallback}
			}

			callbackExecutor := server.NewCallbackExecutor(callbackConfig, logger)

			agent := server.NewOpenAICompatibleAgent(logger)
			agent.SetLLMClient(mockLLMClient)
			agent.SetToolBox(toolBox)
			agent.SetCallbackExecutor(callbackExecutor)

			messages := []types.Message{
				{
					Role: "user",
					Parts: []types.Part{
						map[string]any{
							"kind": "text",
							"text": "Use the test tool",
						},
					},
				},
			}

			result, err := agent.Run(context.Background(), messages)

			require.NoError(t, err)
			assert.NotNil(t, result)

			// Verify tool execution based on callback behavior
			if tt.expectedSkipTool {
				assert.False(t, toolExecuted, "Tool should have been skipped by before callback")
			} else {
				assert.True(t, toolExecuted, "Tool should have been executed")
			}

			// Find tool result in additional messages
			foundExpectedResult := false
			for _, msg := range result.AdditionalMessages {
				if msg.Role == "tool" && len(msg.Parts) > 0 {
					if data, ok := msg.Parts[0].(map[string]any); ok {
						if dataMap, ok := data["data"].(map[string]any); ok {
							if result, ok := dataMap["result"].(string); ok {
								var m map[string]string
								if err := json.Unmarshal([]byte(result), &m); err != nil {
									t.Errorf("failed to unmarshal result: %v", err)
								}

								if m["result"] == tt.expectedToolResult {
									foundExpectedResult = true
									break
								}
							}
						}
					}
				}
			}

			// Debug: print the actual messages if test is failing
			if !foundExpectedResult {
				t.Logf("Expected tool result '%v' not found. Actual additional messages:", result.AdditionalMessages)
				for i, msg := range result.AdditionalMessages {
					t.Logf("Message %d: Role=%s, Parts=%+v", i, msg.Role, msg.Parts)
				}
			}

			assert.True(t, foundExpectedResult, fmt.Sprintf("Expected to find tool result '%s' in additional messages", tt.expectedToolResult))

			// Should have made 2 LLM calls (before and after tool execution)
			assert.Equal(t, 2, mockLLMClient.CreateChatCompletionCallCount())
		})
	}
}
