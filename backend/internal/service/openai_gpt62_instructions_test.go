package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type gpt62InstructionsCase struct {
	name    string
	present bool
	value   any
	want    string
}

var gpt62InstructionsCases = []gpt62InstructionsCase{
	{name: "missing"},
	{name: "null", present: true},
	{name: "empty", present: true, value: ""},
	{name: "whitespace", present: true, value: " \n\t"},
	{name: "explicit", present: true, value: "  Caller instructions.\n", want: "  Caller instructions.\n"},
}

type gpt62ForwardOptions struct {
	accountType    string
	cli            bool
	path           string
	passthrough    bool
	lite           bool
	mapping        map[string]any
	compactMapping map[string]any
	forcedTemplate string
	imageBridge    bool
}

// Capture at the HTTP upstream boundary, after Forward's early default and the
// OAuth transform. A successful simulated completion also exercises the return path.
func captureGPT62InstructionsForward(t *testing.T, payload map[string]any, opts gpt62ForwardOptions) []byte {
	t.Helper()
	gin.SetMode(gin.TestMode)
	body, err := json.Marshal(payload)
	require.NoError(t, err)
	if opts.path == "" {
		opts.path = "/v1/responses"
	}
	if opts.accountType == "" {
		opts.accountType = AccountTypeOAuth
	}
	response := `{"id":"resp_fixture","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}`
	contentType := "text/event-stream"
	if !strings.HasSuffix(opts.path, "/compact") {
		response = "data: {\"type\":\"response.completed\",\"response\":" + response + "}\n\ndata: [DONE]\n\n"
	} else {
		contentType = "application/json"
	}
	upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(strings.NewReader(response))}}
	svc := &OpenAIGatewayService{cfg: &config.Config{Gateway: config.GatewayConfig{ForcedCodexInstructionsTemplate: opts.forcedTemplate, CodexImageGenerationBridgeEnabled: opts.imageBridge}}, httpUpstream: upstream}
	account := &Account{ID: 1, Platform: PlatformOpenAI, Type: opts.accountType, Status: StatusActive, Schedulable: true, Concurrency: 1,
		Credentials: map[string]any{"api_key": "fixture-key", "access_token": "fixture-token", "chatgpt_account_id": "fixture-account", "base_url": "https://upstream.example"},
		Extra:       map[string]any{"openai_responses_supported": true, "openai_passthrough": opts.passthrough, "openai_oauth_responses_websockets_v2_mode": OpenAIWSIngressModeOff},
	}
	if opts.mapping != nil {
		account.Credentials["model_mapping"] = opts.mapping
	}
	if opts.compactMapping != nil {
		account.Credentials["compact_model_mapping"] = opts.compactMapping
	}
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, opts.path, bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("User-Agent", "OpenAI/Python fixture")
	if opts.cli {
		c.Request.Header.Set("User-Agent", "codex_cli_rs/0.98.0")
		c.Request.Header.Set("Originator", "codex_cli_rs")
	}
	if opts.lite {
		c.Request.Header.Set(responsesLiteHeader, "true")
	}
	SetOpenAIClientTransport(c, OpenAIClientTransportHTTP)
	var result *OpenAIForwardResult
	switch opts.path {
	case "/v1/chat/completions":
		result, err = svc.ForwardAsChatCompletions(context.Background(), c, account, body, "", "")
	case "/v1/messages":
		result, err = svc.ForwardAsAnthropic(context.Background(), c, account, body, "", "")
	default:
		result, err = svc.Forward(context.Background(), c, account, body)
	}
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq, "must reach the real outbound request builder")
	require.Len(t, upstream.bodies, 1)
	return upstream.lastBody
}

func requireGPT62Instructions(t *testing.T, wire []byte, want string) {
	t.Helper()
	value := gjson.GetBytes(wire, "instructions")
	require.True(t, value.Exists(), "instructions must be present")
	require.Equal(t, gjson.String, value.Type)
	require.Equal(t, want, value.String())
	requireNoGeneratedCodexTemplate(t, wire)
}

// Inspect decoded strings so escaping or moving a template to a developer
// message cannot hide an accidental default. Template contents may evolve.
func requireNoGeneratedCodexTemplate(t *testing.T, wire []byte) {
	t.Helper()
	var decoded any
	require.NoError(t, json.Unmarshal(wire, &decoded))
	templates := []string{openai.DefaultInstructions}
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-sol", "gpt-6-astra", "gpt-5.1-codex", "gpt-5.5"} {
		template := strings.TrimSpace(openai.CodexBaseInstructionsForModel(model))
		require.NotEmpty(t, template, "catalog must retain full templates for %s", model)
		templates = append(templates, template)
	}
	var visit func(any)
	visit = func(value any) {
		switch value := value.(type) {
		case string:
			for _, template := range templates {
				require.False(t, strings.Contains(value, template), "unexpected generated Codex template")
			}
		case []any:
			for _, item := range value {
				visit(item)
			}
		case map[string]any:
			for _, item := range value {
				visit(item)
			}
		}
	}
	visit(decoded)
}

func TestGPT62InstructionsForwardMatrix(t *testing.T) {
	inputs := []struct {
		name  string
		input any
		count int64
	}{
		{"string", "User input.", 1},
		{"messages", []any{map[string]any{"role": "user", "content": "User input."}}, 1},
		{"developer", []any{map[string]any{"role": "developer", "content": "Caller developer."}, map[string]any{"role": "user", "content": "User input."}}, 2},
	}
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, accountType := range []string{AccountTypeOAuth, AccountTypeSetupToken} {
			for _, cli := range []bool{false, true} {
				for _, input := range inputs {
					for _, instructions := range gpt62InstructionsCases {
						t.Run(fmt.Sprintf("%s/%s/cli=%t/%s/%s", model, accountType, cli, input.name, instructions.name), func(t *testing.T) {
							payload := map[string]any{"model": model, "input": input.input, "stream": true}
							if instructions.present {
								payload["instructions"] = instructions.value
							}
							wire := captureGPT62InstructionsForward(t, payload, gpt62ForwardOptions{accountType: accountType, cli: cli})
							requireGPT62Instructions(t, wire, instructions.want)
							require.Equal(t, model, gjson.GetBytes(wire, "model").String())
							require.Equal(t, input.count, gjson.GetBytes(wire, "input.#").Int())
							require.Equal(t, 1, strings.Count(string(wire), "User input."))
							if input.name == "developer" {
								require.Equal(t, "Caller developer.", gjson.GetBytes(wire, "input.0.content").String())
							}
						})
					}
				}
			}
		}
	}
}

func TestGPT62InstructionsMappedForward(t *testing.T) {
	for _, tc := range []struct{ request, mapped string }{
		{"public-sol", "gpt-6.2-sol"}, {"public-astra", "gpt-6.2-astra"},
		{"gpt-6.2-sol", "gpt-6.2-astra"}, {"gpt-6.2-astra", "gpt-6.2-sol"},
		{"gpt-5.1-codex", "gpt-6.2-sol"}, {"gpt-5.1-codex", "gpt-6.2-astra"},
		{"gpt-6.2-sol", "gpt-5.3-codex"}, {"gpt-6.2-astra", "gpt-5.3-codex"},
	} {
		t.Run(tc.request+"/"+tc.mapped, func(t *testing.T) {
			wire := captureGPT62InstructionsForward(t, map[string]any{"model": tc.request, "input": "hello"}, gpt62ForwardOptions{mapping: map[string]any{tc.request: tc.mapped}})
			require.Equal(t, tc.mapped, gjson.GetBytes(wire, "model").String())
			if strings.HasPrefix(tc.mapped, "gpt-6.2-") {
				requireGPT62Instructions(t, wire, "")
			} else {
				require.Equal(t, strings.TrimSpace(openai.CodexBaseInstructionsForModel(tc.mapped)), gjson.GetBytes(wire, "instructions").String())
			}
		})
	}
}

func TestGPT62InstructionsCompactMapping(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
			for _, instructions := range gpt62InstructionsCases {
				t.Run(fmt.Sprintf("%s/passthrough=%t/%s", model, passthrough, instructions.name), func(t *testing.T) {
					payload := map[string]any{"model": "gpt-5.1-codex", "input": []any{map[string]any{"role": "user", "content": "hello"}}}
					if instructions.present {
						payload["instructions"] = instructions.value
					}
					wire := captureGPT62InstructionsForward(t, payload, gpt62ForwardOptions{path: "/v1/responses/compact", passthrough: passthrough, mapping: map[string]any{"gpt-5.1-codex": "gpt-5.2"}, compactMapping: map[string]any{"gpt-5.1-codex": model}})
					require.Equal(t, model, gjson.GetBytes(wire, "model").String())
					if passthrough {
						// Direct forwarding preserves caller fields after applying the compact mapping.
						got := gjson.GetBytes(wire, "instructions")
						require.Equal(t, instructions.present, got.Exists())
						if instructions.present {
							expected, err := json.Marshal(instructions.value)
							require.NoError(t, err)
							require.JSONEq(t, string(expected), got.Raw)
						}
						requireNoGeneratedCodexTemplate(t, wire)
					} else {
						requireGPT62Instructions(t, wire, instructions.want)
					}
				})
			}
		}
	}
}

func TestGPT62InstructionsAliasesAndIdentity(t *testing.T) {
	for _, base := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		aliases := []string{base, "provider/openai/" + base, strings.ToUpper(base), strings.ReplaceAll(base, "-", "_"), strings.ReplaceAll(base, "-", " "), strings.ReplaceAll(base, "-", "--")}
		for _, suffix := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max", "openai-compact", "2026-11-24"} {
			aliases = append(aliases, base+"-"+suffix)
		}
		for _, alias := range aliases {
			t.Run(alias, func(t *testing.T) {
				require.Equal(t, "", defaultCodexSynthInstructions(alias))
				wire := captureGPT62InstructionsForward(t, map[string]any{"model": alias, "input": "hello", "reasoning": map[string]any{"effort": "minimal"}}, gpt62ForwardOptions{})
				requireGPT62Instructions(t, wire, "")
				require.Equal(t, base, gjson.GetBytes(wire, "model").String())
				require.Equal(t, "none", gjson.GetBytes(wire, "reasoning.effort").String(), "effort field follows its own normalization")
			})
		}
	}
	for _, model := range []string{"gpt-6.2-astra-pro", "openai/GPT-6.2-ASTRA-PRO-XHIGH", "gpt-6.2-astra-pro-2026-11-24", "gpt-6.2-astral", "gpt-6.2-solitude", "gpt-6.20-astra", "gpt-5.1-codex", "gpt-6-astra"} {
		t.Run("non-target/"+model, func(t *testing.T) {
			require.NotEmpty(t, defaultCodexSynthInstructions(model))
			// Direct transform preserves independent model identities before upstream routing.
			body := map[string]any{"model": model, "input": "hello"}
			require.NoError(t, applyCodexOAuthTransformWithOptions(body, codexOAuthTransformOptions{}).Error)
			require.NotEmpty(t, body["instructions"])
			require.Equal(t, model, body["model"])
		})
	}
	require.Empty(t, defaultCodexSynthInstructions("gpt-6.1-sol"))
}

func TestGPT62InstructionsContentAndIdempotence(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, tc := range []struct {
			name, fields, want string
			count              int64
		}{
			{"system", `"input":[{"role":"system","content":"  System guidance.\n"},{"role":"user","content":"hello"}]`, "  System guidance.\n", 1},
			{"system and instructions", `"instructions":"  Explicit.\n","input":[{"role":"system","content":"System guidance."},{"role":"user","content":"hello"}]`, "System guidance.\n\n  Explicit.\n", 1},
			{"json object", `"text":{"format":{"type":"json_object"}},"input":[{"role":"system","content":"Return JSON."},{"role":"user","content":"hello"}]`, "Return JSON.", 2},
			{"json schema", `"text":{"format":{"type":"json_schema","name":"result","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}},"input":[{"role":"user","content":"hello"}]`, "", 1},
			{"mixed system", `"input":[{"role":"system","content":[{"type":"input_text","text":"System image."},{"type":"input_image","image_url":"https://example.com/image.png"}]},{"role":"user","content":"hello"}]`, "System image.", 2},
			{"mixed user", `"input":[{"role":"user","content":[{"type":"input_text","text":"hello"},{"type":"input_image","image_url":"https://example.com/image.png"}]}]`, "", 1},
			{"legacy prompt", `"prompt":"hello"`, "", 1},
			{"legacy prompt array", `"prompt":[{"role":"user","content":"hello"}]`, "", 1},
			{"tool history", `"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"input":[{"role":"user","content":"original question"},{"type":"function_call","call_id":"call_fixture","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"call_fixture","output":"recorded result"},{"role":"assistant","content":"recorded answer"},{"role":"user","content":"hello"}]`, "", 5},
		} {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				var payload map[string]any
				require.NoError(t, json.Unmarshal([]byte(`{"model":"`+model+`",`+tc.fields+`}`), &payload))
				wire := captureGPT62InstructionsForward(t, payload, gpt62ForwardOptions{})
				requireGPT62Instructions(t, wire, tc.want)
				require.Equal(t, tc.count, gjson.GetBytes(wire, "input.#").Int())
				if strings.HasPrefix(tc.name, "mixed") {
					require.Contains(t, string(wire), "https://example.com/image.png")
				}
				if tc.name == "json object" {
					require.Equal(t, "developer", gjson.GetBytes(wire, "input.0.role").String())
					require.Contains(t, gjson.GetBytes(wire, "input.0.content").Raw, "Return JSON.")
				}
				if tc.name == "json schema" {
					require.Equal(t, "json_schema", gjson.GetBytes(wire, "text.format.type").String())
					require.Equal(t, "string", gjson.GetBytes(wire, "text.format.schema.properties.answer.type").String())
				}
				if tc.name == "tool history" {
					require.Equal(t, "original question", gjson.GetBytes(wire, "input.0.content").String())
					require.Equal(t, "function_call", gjson.GetBytes(wire, "input.1.type").String())
					require.Equal(t, "function_call_output", gjson.GetBytes(wire, "input.2.type").String())
					require.Equal(t, gjson.GetBytes(wire, "input.1.call_id").String(), gjson.GetBytes(wire, "input.2.call_id").String())
					require.NotEmpty(t, gjson.GetBytes(wire, "input.1.call_id").String())
					require.Equal(t, "recorded result", gjson.GetBytes(wire, "input.2.output").String())
					require.Equal(t, "recorded answer", gjson.GetBytes(wire, "input.3.content").String())
				}
				var transformed map[string]any
				require.NoError(t, json.Unmarshal(wire, &transformed))
				require.False(t, applyInstructions(transformed, false))
				for range 2 {
					require.NoError(t, applyCodexOAuthTransformWithOptions(transformed, codexOAuthTransformOptions{}).Error)
					again, err := json.Marshal(transformed)
					require.NoError(t, err)
					require.JSONEq(t, string(wire), string(again), "repeated transforms must preserve content and instructions")
				}
			})
		}
		for _, cli := range []bool{false, true} {
			for _, tc := range gpt62InstructionsCases {
				t.Run(fmt.Sprintf("%s/apply/cli=%t/%s", model, cli, tc.name), func(t *testing.T) {
					body := map[string]any{"model": model}
					if tc.present {
						body["instructions"] = tc.value
					}
					applyInstructions(body, cli)
					require.Equal(t, tc.want, body["instructions"])
					require.False(t, applyInstructions(body, cli))
				})
			}
		}
	}
}

func TestGPT62InstructionsCallerTemplatesAreNotRemoved(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		template := openai.CodexBaseInstructionsForModel(model)
		require.Greater(t, len(template), 2000, "model catalog continues to expose a full base template")
		descriptor := newConfiguredCodexModelDescriptor(model)
		require.Equal(t, template, descriptor.ModelMessages.InstructionsTemplate)
		for _, source := range []string{"instructions", "developer", "system"} {
			t.Run(model+"/"+source, func(t *testing.T) {
				input := []any{map[string]any{"role": "user", "content": "hello"}}
				payload := map[string]any{"model": model, "input": input}
				if source == "instructions" {
					payload["instructions"] = template
				} else {
					payload["input"] = append([]any{map[string]any{"role": source, "content": template}}, input...)
				}
				wire := captureGPT62InstructionsForward(t, payload, gpt62ForwardOptions{})
				if source == "developer" {
					require.Equal(t, template, gjson.GetBytes(wire, "input.0.content").String())
					require.Equal(t, "", gjson.GetBytes(wire, "instructions").String())
				} else {
					require.Equal(t, template, gjson.GetBytes(wire, "instructions").String())
					require.Equal(t, int64(1), gjson.GetBytes(wire, "input.#").Int())
				}
			})
		}
	}
}

func TestGPT62InstructionsBridgesAndLite(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, tc := range []struct {
			name, path, fields, want string
			lite                     bool
		}{
			{"chat", "/v1/chat/completions", `"messages":[{"role":"user","content":"hello"}]`, "", false},
			{"chat system", "/v1/chat/completions", `"messages":[{"role":"system","content":"Caller guidance."},{"role":"user","content":"hello"}]`, "Caller guidance.", false},
			{"chat responses shape", "/v1/chat/completions", `"input":"hello"`, "", false},
			{"messages", "/v1/messages", `"max_tokens":100,"messages":[{"role":"user","content":"hello"}]`, "", false},
			{"messages system", "/v1/messages", `"max_tokens":100,"system":"Caller guidance.","messages":[{"role":"user","content":"hello"}]`, "", false},
			{"lite", "/v1/responses", `"input":[{"type":"additional_tools","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]},{"type":"message","role":"developer","content":[{"type":"input_text","text":"Caller developer."}]},{"role":"user","content":"hello"}]`, "", true},
		} {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				var payload map[string]any
				require.NoError(t, json.Unmarshal([]byte(`{"model":"`+model+`",`+tc.fields+`}`), &payload))
				wire := captureGPT62InstructionsForward(t, payload, gpt62ForwardOptions{path: tc.path, lite: tc.lite, cli: tc.lite})
				requireGPT62Instructions(t, wire, tc.want)
				require.Contains(t, string(wire), "hello")
				if tc.name == "messages system" {
					require.Equal(t, "developer", gjson.GetBytes(wire, "input.0.role").String())
					require.Contains(t, gjson.GetBytes(wire, "input.0.content").Raw, "Caller guidance.")
				}
				if tc.lite {
					require.Equal(t, "additional_tools", gjson.GetBytes(wire, "input.0.type").String())
					require.Equal(t, "lookup", gjson.GetBytes(wire, "input.0.tools.0.name").String())
					require.Equal(t, "developer", gjson.GetBytes(wire, "input.1.role").String())
					require.Equal(t, "Caller developer.", gjson.GetBytes(wire, "input.1.content.0.text").String())
					require.Equal(t, int64(3), gjson.GetBytes(wire, "input.#").Int())
				}
			})
		}
	}
}

func TestGPT62InstructionsDirectForwardPreservesFields(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, accountType := range []string{AccountTypeOAuth, AccountTypeAPIKey} {
			for _, tc := range gpt62InstructionsCases {
				t.Run(model+"/"+accountType+"/"+tc.name, func(t *testing.T) {
					payload := map[string]any{"model": model, "input": []any{map[string]any{"role": "developer", "content": "Caller developer."}, map[string]any{"role": "user", "content": "hello"}}}
					if tc.present {
						payload["instructions"] = tc.value
					}
					wire := captureGPT62InstructionsForward(t, payload, gpt62ForwardOptions{accountType: accountType, passthrough: accountType == AccountTypeOAuth})
					require.Equal(t, tc.present, gjson.GetBytes(wire, "instructions").Exists())
					if tc.present {
						expected, err := json.Marshal(tc.value)
						require.NoError(t, err)
						require.JSONEq(t, string(expected), gjson.GetBytes(wire, "instructions").Raw)
					}
					input, err := json.Marshal(payload["input"])
					require.NoError(t, err)
					require.JSONEq(t, string(input), gjson.GetBytes(wire, "input").Raw)
					requireNoGeneratedCodexTemplate(t, wire)
				})
			}
		}
	}
}

func TestGPT62InstructionsExplicitFeatures(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		t.Run(model+"/administrator template", func(t *testing.T) {
			wire := captureGPT62InstructionsForward(t, map[string]any{"model": model, "system": "Caller guidance.", "max_tokens": 100, "messages": []any{map[string]any{"role": "user", "content": "hello"}}}, gpt62ForwardOptions{path: "/v1/messages", forcedTemplate: "Admin prefix.\n\n{{ .ExistingInstructions }}"})
			requireGPT62Instructions(t, wire, "Admin prefix.\n\nCaller guidance.")
		})
		t.Run(model+"/image bridge", func(t *testing.T) {
			wire := captureGPT62InstructionsForward(t, map[string]any{"model": model, "input": "Draw a cat."}, gpt62ForwardOptions{cli: true, imageBridge: true})
			requireNoGeneratedCodexTemplate(t, wire)
			instructions := gjson.GetBytes(wire, "instructions").String()
			require.Contains(t, instructions, codexImageGenerationBridgeMarker)
			require.Contains(t, string(wire), "image_generation")
			var decoded map[string]any
			require.NoError(t, json.Unmarshal(wire, &decoded))
			require.False(t, applyCodexImageGenerationBridgeInstructions(decoded))
			require.Equal(t, instructions, decoded["instructions"])
		})
	}
}

func TestGPT62InstructionsCompactMappingToOlderModel(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, passthrough := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/passthrough=%t", model, passthrough), func(t *testing.T) {
				wire := captureGPT62InstructionsForward(t, map[string]any{"model": model, "input": []any{map[string]any{"role": "user", "content": "hello"}}}, gpt62ForwardOptions{path: "/v1/responses/compact", passthrough: passthrough, compactMapping: map[string]any{model: "gpt-5.3-codex"}})
				require.Equal(t, "gpt-5.3-codex", gjson.GetBytes(wire, "model").String())
				require.Equal(t, strings.TrimSpace(openai.CodexBaseInstructionsForModel("gpt-5.3-codex")), gjson.GetBytes(wire, "instructions").String())
			})
		}
	}
}

func TestGPT62InstructionsReasoningEffortIndependent(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, effort := range []string{"low", "medium", "high", "xhigh", "max", "ultra"} {
			t.Run(model+"/"+effort, func(t *testing.T) {
				wire := captureGPT62InstructionsForward(t, map[string]any{"model": model + "-high", "input": "hello", "reasoning": map[string]any{"effort": effort}}, gpt62ForwardOptions{})
				requireGPT62Instructions(t, wire, "")
				require.Equal(t, model, gjson.GetBytes(wire, "model").String())
				require.Equal(t, effort, gjson.GetBytes(wire, "reasoning.effort").String())
			})
		}
	}
}

func TestGPT62InstructionsPassthroughProviderAliases(t *testing.T) {
	for _, model := range []string{"codex/gpt-6.2-sol", "codex/GPT_6.2_ASTRA-2026-11-24"} {
		for _, tc := range gpt62InstructionsCases {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				payload := map[string]any{"model": model, "input": "hello"}
				if tc.present {
					payload["instructions"] = tc.value
				}
				wire := captureGPT62InstructionsForward(t, payload, gpt62ForwardOptions{passthrough: true})
				require.Equal(t, model, gjson.GetBytes(wire, "model").String(), "passthrough preserves model spelling")
				require.Equal(t, tc.present, gjson.GetBytes(wire, "instructions").Exists())
				if tc.present {
					expected, err := json.Marshal(tc.value)
					require.NoError(t, err)
					require.JSONEq(t, string(expected), gjson.GetBytes(wire, "instructions").Raw)
				}
				requireNoGeneratedCodexTemplate(t, wire)
			})
		}
	}
}
