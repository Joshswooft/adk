package server

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	sdk "github.com/inference-gateway/sdk"
)

func TestNewDefaultToolBox_IncludesInputRequiredTool(t *testing.T) {
	toolBox := NewDefaultToolBox()

	if !toolBox.HasTool("input_required") {
		t.Error("Expected default toolbox to include 'input_required' tool")
	}

	toolNames := toolBox.GetToolNames()
	found := false
	for _, name := range toolNames {
		if name == "input_required" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected 'input_required' to be in tool names list")
	}

	result, err := toolBox.ExecuteTool(context.Background(), "input_required", map[string]any{
		"message": "Please provide more details about your request",
	})
	if err != nil {
		t.Errorf("Expected no error when executing input_required tool, got: %v", err)
	}

	expectedResult := "Input requested from user: Please provide more details about your request"
	if result != expectedResult {
		t.Errorf("Expected result '%s', got '%s'", expectedResult, result)
	}
}

func TestNewDefaultToolBox_GetTools(t *testing.T) {
	toolBox := NewDefaultToolBox()
	tools := toolBox.GetTools()

	if len(tools) == 0 {
		t.Error("Expected at least one tool in default toolbox")
	}

	var inputRequiredTool *sdk.FunctionObject
	for _, tool := range tools {
		if tool.Function.Name == "input_required" {
			inputRequiredTool = &tool.Function
			break
		}
	}

	if inputRequiredTool == nil {
		t.Error("Expected to find input_required tool in GetTools() result")
		return
	}

	if inputRequiredTool.Description == nil || *inputRequiredTool.Description == "" {
		t.Error("Expected input_required tool to have a description")
	}

	if inputRequiredTool.Parameters == nil {
		t.Error("Expected input_required tool to have parameters")
	}
}

// helper to compare tools
var toolComparer = func(x, y Tool) bool {
	return x.GetName() == y.GetName() && x.GetDescription() == y.GetDescription()
}

func TestNewDefaultToolBox_GetTool(t *testing.T) {

	tests := map[string]struct {
		setupToolbox  func() ToolBox
		toolToFind    string
		expectedTool  Tool
		expectedFound bool
	}{
		"Returns false when the tool does not exist in the toolbox": {
			toolToFind:    "test",
			expectedFound: false,
			expectedTool:  nil,
			setupToolbox: func() ToolBox {
				return NewDefaultToolBox()
			},
		},
		"Returns the tool in the toolbox with the matching tool name": {
			toolToFind:    "test",
			expectedTool:  NewBasicTool("test", "a test tool", nil, nil),
			expectedFound: true,
			setupToolbox: func() ToolBox {
				toolbox := NewDefaultToolBox()
				toolbox.AddTool(NewBasicTool("test", "a test tool", nil, nil))
				return toolbox
			},
		},
	}

	for name, tc := range tests {
		gotTool, gotFound := tc.setupToolbox().GetTool(tc.toolToFind)
		if gotFound != tc.expectedFound {
			t.Errorf("testCase: '%s', Expected found '%t' not equal to got found: '%t'", name, tc.expectedFound, gotFound)
			return
		}

		if diff := cmp.Diff(gotTool, tc.expectedTool, cmp.Comparer(toolComparer)); diff != "" {
			t.Errorf("testCase: '%s': GetTool() mismatch (-want +got):\n%s", name, diff)
		}
	}

}

func TestNewToolBox_CreatesEmptyToolBox(t *testing.T) {
	toolBox := NewToolBox()

	if toolBox == nil {
		t.Error("Expected NewToolBox to return a non-nil toolbox")
		return
	}

	toolNames := toolBox.GetToolNames()
	if len(toolNames) != 0 {
		t.Errorf("Expected empty toolbox to have 0 tools, got %d", len(toolNames))
	}

	tools := toolBox.GetTools()
	if len(tools) != 0 {
		t.Errorf("Expected empty toolbox to return 0 tools from GetTools(), got %d", len(tools))
	}

	if toolBox.HasTool("input_required") {
		t.Error("Expected empty toolbox to not have any tools, including input_required")
	}

	testTool := NewBasicTool(
		"test_tool",
		"A test tool",
		map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
		func(ctx context.Context, args map[string]any) (string, error) {
			return "test result", nil
		},
	)

	toolBox.AddTool(testTool)

	if !toolBox.HasTool("test_tool") {
		t.Error("Expected to be able to add tool to empty toolbox")
	}

	if len(toolBox.GetToolNames()) != 1 {
		t.Errorf("Expected toolbox to have 1 tool after adding, got %d", len(toolBox.GetToolNames()))
	}
}
