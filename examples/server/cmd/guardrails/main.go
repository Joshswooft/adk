package guardrails

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/inference-gateway/adk/server"
	"github.com/inference-gateway/adk/server/config"
	"github.com/inference-gateway/adk/types"
	"github.com/sethvargo/go-envconfig"
	"go.uber.org/zap"
)

func main() {
	fmt.Println("🤖 Starting ADK Callback Example...")

	// Initialize logger
	logger, err := zap.NewDevelopment()
	if err != nil {
		log.Fatalf("failed to create logger: %v", err)
	}
	defer logger.Sync()

	// Create agent with callbacks using the builder pattern
	agent, err := server.NewAgentBuilder(logger).
		WithBeforeModelCallback(simpleBeforeModelGuardrail).
		WithAfterModelCallback(simpleAfterModelModifier).
		WithSystemPrompt("You are a helpful AI assistant. Be concise and friendly in your responses.").
		Build()

	if err != nil {
		logger.Fatal("failed to create agent", zap.Error(err))
	}

	// Step 2: Load configuration from environment
	cfg := config.Config{
		AgentName:        "agent-with-guardrails",
		AgentDescription: "An AI-powered agent that can be interupted by its configured guard rails",
		AgentVersion:     "1.0.0",
		CapabilitiesConfig: config.CapabilitiesConfig{
			Streaming:              false, // Disable streaming for this background-focused example
			PushNotifications:      true,
			StateTransitionHistory: false,
		},
		QueueConfig: config.QueueConfig{
			CleanupInterval: 5 * time.Minute,
		},
		ServerConfig: config.ServerConfig{
			Port: "8080",
		},
	}

	ctx := context.Background()
	if err := envconfig.Process(ctx, &cfg); err != nil {
		logger.Fatal("failed to process environment config", zap.Error(err))
	}

	// Create and start server with default background task handler

	a2aServer, err := server.NewA2AServerBuilder(cfg, logger).
		WithAgent(agent).
		WithDefaultTaskHandlers().
		WithAgentCard(types.AgentCard{
			Name:        cfg.AgentName,
			Description: cfg.AgentDescription,
			URL:         cfg.AgentURL,
			Version:     cfg.AgentVersion,
			Capabilities: types.AgentCapabilities{
				Streaming:              &cfg.CapabilitiesConfig.Streaming,
				PushNotifications:      &cfg.CapabilitiesConfig.PushNotifications,
				StateTransitionHistory: &cfg.CapabilitiesConfig.StateTransitionHistory,
			},
			DefaultInputModes:  []string{"text/plain"},
			DefaultOutputModes: []string{"text/plain"},
		}).
		Build()

	if err != nil {
		logger.Fatal("failed to create A2A server", zap.Error(err))
	}

	// Start server
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := a2aServer.Start(ctx); err != nil {
			logger.Fatal("server failed to start", zap.Error(err))
		}
	}()

	// Wait for shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("🛑 shutting down server...")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := a2aServer.Stop(shutdownCtx); err != nil {
		logger.Error("shutdown error", zap.Error(err))
	} else {
		logger.Info("✅ goodbye!")
	}
}

// simpleBeforeModelGuardrail demonstrates blocking LLM calls based on content
func simpleBeforeModelGuardrail(ctx context.Context, callbackContext *server.CallbackContext, llmRequest *server.LLMRequest) *server.LLMResponse {
	logger := callbackContext.Logger

	logger.Info("before model callback triggered",
		zap.String("agent_name", callbackContext.AgentName),
		zap.String("invocation_id", callbackContext.InvocationID))

	// Check if the last user message contains blocked content
	if len(llmRequest.Contents) > 0 {
		lastMessage := llmRequest.Contents[len(llmRequest.Contents)-1]
		if lastMessage.Role == "user" && len(lastMessage.Parts) > 0 {
			if textPart, ok := lastMessage.Parts[0].(map[string]any); ok {
				if text, exists := textPart["text"].(string); exists {
					logger.Info("inspecting user message", zap.String("text", text))

					// Block messages containing "BLOCK" (case-insensitive)
					if strings.Contains(strings.ToUpper(text), "BLOCK") {
						logger.Info("blocked content detected, skipping LLM call")

						// Return a response instead of calling the LLM
						return &server.LLMResponse{
							Content: &types.Message{
								Kind:      "message",
								MessageID: fmt.Sprintf("blocked-%d", callbackContext.InvocationID),
								Role:      "assistant",
								Parts: []types.Part{
									map[string]any{
										"kind": "text",
										"text": "I'm sorry, but I cannot process messages containing blocked content. Please rephrase your request.",
									},
								},
							},
						}
					}

					// Add a prefix to system instruction to modify LLM behavior
					if llmRequest.Config != nil && strings.Contains(strings.ToLower(text), "modify") {
						logger.Info("modifying system instruction for special request")

						prefix := "[Modified by Callback] "
						if llmRequest.Config.SystemInstruction != nil {
							// Modify existing system instruction
							if len(llmRequest.Config.SystemInstruction.Parts) > 0 {
								if textPart, ok := llmRequest.Config.SystemInstruction.Parts[0].(map[string]any); ok {
									if originalText, exists := textPart["text"].(string); exists {
										textPart["text"] = prefix + originalText
									}
								}
							}
						} else {
							// Add new system instruction
							llmRequest.Config.SystemInstruction = &types.Message{
								Role: "system",
								Parts: []types.Part{
									map[string]any{
										"kind": "text",
										"text": prefix + "You are a helpful assistant. Always acknowledge when your behavior has been modified by a callback.",
									},
								},
							}
						}
					}
				}
			}
		}
	}

	logger.Info("allowing LLM call to proceed")
	return nil // Allow the LLM call to proceed
}

// simpleAfterModelModifier demonstrates modifying LLM responses
func simpleAfterModelModifier(ctx context.Context, callbackContext *server.CallbackContext, llmResponse *server.LLMResponse) *server.LLMResponse {
	logger := callbackContext.Logger

	logger.Info("after model callback triggered")

	// Check if response contains "joke" and replace it with "funny story"
	if llmResponse.Content != nil && len(llmResponse.Content.Parts) > 0 {
		if textPart, ok := llmResponse.Content.Parts[0].(map[string]any); ok {
			if text, exists := textPart["text"].(string); exists {
				if strings.Contains(strings.ToLower(text), "joke") {
					logger.Info("modifying response: replacing 'joke' with 'funny story'")

					// Create a modified response
					modifiedText := strings.ReplaceAll(text, "joke", "funny story")
					modifiedText = strings.ReplaceAll(modifiedText, "Joke", "Funny story")

					return &server.LLMResponse{
						Content: &types.Message{
							Kind:      llmResponse.Content.Kind,
							MessageID: llmResponse.Content.MessageID,
							Role:      llmResponse.Content.Role,
							Parts: []types.Part{
								map[string]any{
									"kind": "text",
									"text": modifiedText,
								},
							},
						},
						GroundingMetadata: llmResponse.GroundingMetadata,
					}
				}
			}
		}
	}

	logger.Info("using original LLM response")
	return nil // Use original response
}
