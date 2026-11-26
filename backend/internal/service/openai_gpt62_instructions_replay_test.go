package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func TestGPT62InstructionsOAuthReplay(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, mode := range []string{"missing", "null", "empty", "whitespace", "system", "explicit", "insert instructions", "replace instructions", "remove tools", "alter input", "alter final instructions"} {
			t.Run(model+"/"+mode, func(t *testing.T) {
				upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_replay\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\ndata: [DONE]\n\n"))}}
				svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream, settingService: NewSettingService(&codexPolicyMigrationRepoStub{values: map[string]string{SettingKeyOpenAI409OmitEncryptedHistoryEnabled: "true"}}, nil)}
				source, err := sjson.Set(issue715OpaqueBody, "model", model)
				require.NoError(t, err)
				c, _ := newTurnStateTestContext(t, 715, "fixture-session")
				c.Set("api_key", &APIKey{ID: 715, GroupID: int64Ptr(715)})
				SetOpenAIClientTransport(c, OpenAIClientTransportHTTP)
				// Learn the opaque source under the same model/session as the later replay.
				learned := svc.WithOpenAIAgentMessageRecovery(context.Background(), c, []byte(source), "fixture-session", model)
				originalReq, err := http.NewRequest(http.MethodPost, "https://native.example/v1/responses", strings.NewReader(source))
				require.NoError(t, err)
				complete, err := svc.prepareOpenAIAgentRecoveryAttempt(learned, &Account{ID: 1, Platform: PlatformOpenAI, Type: AccountTypeAPIKey}, originalReq)
				require.NoError(t, err)
				complete(true)
				body := strings.TrimSuffix(source, "]}") + `,{"type":"function_call","call_id":"call_recorded","name":"record","arguments":"{}"},{"type":"function_call_output","call_id":"call_recorded","output":"completed once"},{"role":"user","content":"finish using the recorded result"}],"tools":[{"type":"function","name":"record","parameters":{"type":"object"}}]}`
				want := ""
				switch mode {
				case "null":
					body, err = sjson.Set(body, "instructions", nil)
				case "empty":
					body, err = sjson.Set(body, "instructions", "")
				case "whitespace":
					body, err = sjson.Set(body, "instructions", " \n\t")
				case "system":
					body, err = sjson.SetRaw(body, "input.-1", `{"role":"system","content":"Keep recorded results."}`)
					want = "Keep recorded results."
				case "explicit", "replace instructions":
					body, err = sjson.Set(body, "instructions", "  Keep recorded results.\n")
					want = "  Keep recorded results.\n"
				}
				require.NoError(t, err)
				ctx := svc.WithOpenAIAgentMessageRecovery(context.Background(), c, []byte(body), "fixture-session", model)
				next, clean, changed, err := svc.TryRebuildOpenAI409History(ctx, c, []byte(body), "fixture-session", model, openAIAgentRecoveryError("owner_unavailable"))
				require.NoError(t, err)
				require.True(t, changed)
				recovery := openAIAgentRecoveryFromContext(next)
				inputDigest, controlsDigest := recovery.inputDigest, recovery.replayControlsDigest
				var decoded map[string]any
				require.NoError(t, decodeOpenAIJSONUseNumber(clean, &decoded))
				if mode == "insert instructions" || mode == "replace instructions" {
					decoded["instructions"] = "Repeat the already completed operation"
					require.Error(t, prepareOpenAI409OAuthReplay(next, c, decoded))
					require.False(t, recovery.oauthReplayPrepared)
					require.Equal(t, inputDigest, recovery.inputDigest)
					require.Equal(t, controlsDigest, recovery.replayControlsDigest)
					require.Empty(t, upstream.requests)
					return
				}
				require.NoError(t, prepareOpenAI409OAuthReplay(next, c, decoded))
				require.Equal(t, inputDigest, recovery.inputDigest)
				require.Equal(t, controlsDigest, recovery.replayControlsDigest)
				require.Equal(t, want, decoded["instructions"])
				for range 2 {
					require.NoError(t, applyCodexOAuthTransformWithOptions(decoded, codexOAuthTransformOptions{PreserveReadableReplay: true}).Error)
				}
				wire, err := json.Marshal(decoded)
				require.NoError(t, err)
				requireGPT62Instructions(t, wire, want)
				switch mode {
				case "remove tools":
					wire, err = sjson.DeleteBytes(wire, "tools")
				case "alter input":
					wire, err = sjson.SetBytes(wire, "input.0.content", "altered history")
				case "alter final instructions":
					wire, err = sjson.SetBytes(wire, "instructions", "Repeat operation")
				}
				require.NoError(t, err)
				req, err := http.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", bytes.NewReader(wire))
				require.NoError(t, err)
				account := &Account{ID: 2, GroupIDs: []int64{715}, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: StatusActive, Schedulable: true, Concurrency: 1, Credentials: map[string]any{"access_token": "fixture-token"}, Extra: map[string]any{"openai_oauth_responses_websockets_v2_mode": OpenAIWSIngressModeOff}}
				finish, err := svc.prepareOpenAIAgentRecoveryAttempt(next, account, req)
				finish(false)
				if mode == "remove tools" || mode == "alter input" || mode == "alter final instructions" {
					require.Error(t, err)
					require.Empty(t, upstream.requests)
					return
				}
				require.NoError(t, err)
				// The ordinary Forward path prepares and validates the replay again before send.
				result, err := svc.Forward(next, c, account, clean)
				require.NoError(t, err)
				require.NotNil(t, result)
				require.Len(t, upstream.bodies, 1)
				requireGPT62Instructions(t, upstream.lastBody, want)
				require.Contains(t, string(upstream.lastBody), "completed once")
				require.Equal(t, "record", gjson.GetBytes(upstream.lastBody, "tools.0.name").String())
			})
		}
	}
}
