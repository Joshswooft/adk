# Guardrails A2A Example

This example demonstrates an A2A server with **callback-based guardrails** that can intercept, modify, or block LLM requests and responses before and after model inference.

## What This Example Shows

- **Before Model Callbacks**: Block or modify requests before they reach the LLM
- **After Model Callbacks**: Modify responses after LLM generation
- **Content Filtering**: Block messages containing specific keywords
- **Response Modification**: Transform LLM responses based on content
- **System Prompt Modification**: Dynamically alter system instructions
- **Agent Builder Pattern**: Configure agents with multiple callback layers

## Key Features

- **Request Blocking**: Messages containing "BLOCK" are intercepted and never reach the LLM
- **Content Modification**: System prompts are modified when user messages contain "modify"
- **Response Transformation**: LLM responses containing "joke" are automatically changed to "funny story"
- **Logging & Observability**: All callback actions are logged for debugging
- **Production Ready**: Follows the same configuration patterns as production agents

## Architecture

The server uses callback functions to implement guardrails:

```go
agent, err := server.NewAgentBuilder(logger).
    WithBeforeModelCallback(simpleBeforeModelGuardrail).
    WithAfterModelCallback(simpleAfterModelModifier).
    WithSystemPrompt("You are a helpful AI assistant. Be concise and friendly in your responses.").
    Build()
```

### Callback Types

1. **Before Model Callback (`simpleBeforeModelGuardrail`)**:
   - Inspects incoming user messages
   - Blocks content containing "BLOCK" keyword
   - Modifies system instructions for messages containing "modify"
   - Returns early responses to bypass LLM entirely

2. **After Model Callback (`simpleAfterModelModifier`)**:
   - Modifies LLM responses after generation
   - Replaces "joke" with "funny story" in responses
   - Maintains original message structure and metadata

## Running the Example

### Prerequisites

- Go 1.25 or later
- An LLM provider API key (OpenAI, Anthropic, etc.)
- Docker and Docker Compose (optional)

### Using Docker Compose (Recommended)

1. **Copy environment variables:**

   ```bash
   cp .env.example .env
   ```

2. **Configure your LLM provider:**

   ```bash
   # Choose your provider
   A2A_AGENT_CLIENT_PROVIDER=openai
   A2A_AGENT_CLIENT_MODEL=gpt-4o-mini
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the example:**

   ```bash
   docker-compose up --build
   ```

### Running Locally

1. **Start the server:**

   ```bash
   cd server
   export A2A_AGENT_CLIENT_PROVIDER=openai
   export A2A_AGENT_CLIENT_MODEL=gpt-4o-mini
   export OPENAI_API_KEY=your_api_key_here
   go run main.go
   ```

2. **Test with a client** (or use curl/Postman to send A2A messages)

## Server Configuration

The server uses environment variables with the `A2A_` prefix:

| Environment Variable         | Description                    | Default                              |
| ---------------------------- | ------------------------------ | ------------------------------------ |
| `A2A_AGENT_NAME`             | Agent name                     | `agent-with-guardrails`              |
| `A2A_AGENT_DESCRIPTION`      | Agent description              | AI-powered agent with guard rails    |
| `A2A_AGENT_VERSION`          | Agent version                  | `1.0.0`                              |
| `A2A_SERVER_PORT`            | Server port                    | `8080`                               |
| `A2A_DEBUG`                  | Enable debug logging           | `false`                              |
| `A2A_CAPABILITIES_STREAMING` | Enable streaming support       | `false`                              |
| `A2A_AGENT_CLIENT_PROVIDER`  | LLM provider                   | Required                             |
| `A2A_AGENT_CLIENT_MODEL`     | Model name                     | Required                             |
| `A2A_AGENT_CLIENT_BASE_URL`  | LLM API endpoint               | Provider default                     |

## Testing Guardrails

### Content Blocking

Send a message containing "BLOCK" (case-insensitive):

```json
{
  "message": {
    "parts": [{"kind": "text", "text": "Please BLOCK this message"}]
  }
}
```

**Expected Response:**
```
"I'm sorry, but I cannot process messages containing blocked content. Please rephrase your request."
```

The LLM is never called, and the blocking is logged.

### System Prompt Modification

Send a message containing "modify":

```json
{
  "message": {
    "parts": [{"kind": "text", "text": "Please modify your behavior"}]
  }
}
```

**Expected Behavior:**
- The system prompt is prefixed with "[Modified by Callback]"
- The LLM receives the modified instructions
- The response acknowledges the modification

### Response Transformation

Ask the LLM to tell a joke:

```json
{
  "message": {
    "parts": [{"kind": "text", "text": "Tell me a funny joke"}]
  }
}
```

**Expected Behavior:**
- The LLM generates a response containing "joke"
- The after-model callback automatically replaces "joke" with "funny story"
- The user receives the modified response

## Understanding the Code

### Before Model Guardrail (`simpleBeforeModelGuardrail`)

```go
func simpleBeforeModelGuardrail(ctx context.Context, callbackContext *server.CallbackContext, llmRequest *server.LLMRequest) *server.LLMResponse {
    // Check if the last user message contains blocked content
    if strings.Contains(strings.ToUpper(text), "BLOCK") {
        // Return a response instead of calling the LLM
        return &server.LLMResponse{
            Content: &types.Message{
                // Custom blocked content response
            }
        }
    }
    
    // Modify system instruction for special requests
    if strings.Contains(strings.ToLower(text), "modify") {
        // Modify the system prompt
        llmRequest.Config.SystemInstruction = modifiedInstruction
    }
    
    return nil // Allow the LLM call to proceed
}
```

### After Model Modifier (`simpleAfterModelModifier`)

```go
func simpleAfterModelModifier(ctx context.Context, callbackContext *server.CallbackContext, llmResponse *server.LLMResponse) *server.LLMResponse {
    // Check if response contains "joke" and replace it
    if strings.Contains(strings.ToLower(text), "joke") {
        modifiedText := strings.ReplaceAll(text, "joke", "funny story")
        
        return &server.LLMResponse{
            Content: &types.Message{
                // Return modified response
            }
        }
    }
    
    return nil // Use original response
}
```

### Agent Builder Configuration

```go path=/Users/joshdando/projects/forks/adk/examples/server/cmd/guardrails/main.go start=31
agent, err := server.NewAgentBuilder(logger).
    WithBeforeModelCallback(simpleBeforeModelGuardrail).
    WithAfterModelCallback(simpleAfterModelModifier).
    WithSystemPrompt("You are a helpful AI assistant. Be concise and friendly in your responses.").
    Build()
```

## Use Cases

This pattern is useful for:

- **Content Safety**: Block harmful, inappropriate, or policy-violating content
- **Compliance**: Ensure responses meet regulatory or business requirements  
- **Brand Guidelines**: Modify language to match brand voice and tone
- **Cost Control**: Prevent expensive LLM calls for certain types of requests
- **A/B Testing**: Dynamically modify prompts or responses for experimentation
- **Content Transformation**: Standardize terminology or format across responses
- **Debugging**: Log and inspect all requests and responses in detail

## Next Steps

- Try the `ai-powered` example for basic AI integration without guardrails
- Check the `streaming` example for real-time responses with callback support
- Explore the `minimal` example for basic A2A concepts
- Implement your own custom guardrails based on your specific requirements

## Production Considerations

- **Error Handling**: Implement proper error handling in callback functions
- **Logging**: Use structured logging for observability and debugging
- **Testing**: Write comprehensive tests for all guardrail scenarios