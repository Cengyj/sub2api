package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

const gpt62InstructionsCompleted = `{"type":"response.completed","response":{"id":"resp_fixture","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}`

func TestGPT62InstructionsHTTPToWebSocket(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, tc := range gpt62InstructionsCases {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				capture := &openAIWSCaptureConn{events: [][]byte{[]byte(gpt62InstructionsCompleted)}}
				upstream := &httpUpstreamRecorder{}
				svc := wsSSETestService(t, &wsSSETestDialer{conn: capture}, upstream)
				account := wsSSETestAccount()
				account.Credentials["model_mapping"] = map[string]any{"public-62": model}
				c, _ := gin.CreateTestContext(httptest.NewRecorder())
				c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
				SetOpenAIClientTransport(c, OpenAIClientTransportHTTP)
				payload := map[string]any{"model": "public-62", "stream": true, "input": []any{map[string]any{"role": "developer", "content": "Caller developer."}, map[string]any{"role": "user", "content": "hello"}}}
				if tc.present {
					payload["instructions"] = tc.value
				}
				body, err := json.Marshal(payload)
				require.NoError(t, err)
				result, err := svc.Forward(context.Background(), c, account, body)
				require.NoError(t, err)
				require.NotNil(t, result)
				require.True(t, result.OpenAIWSMode)
				require.Empty(t, upstream.requests, "must exercise WS, not fallback HTTP")
				capture.mu.Lock()
				wire, err := json.Marshal(capture.lastWrite)
				writes := len(capture.writes)
				capture.mu.Unlock()
				require.NoError(t, err)
				require.Equal(t, 1, writes)
				requireGPT62Instructions(t, wire, tc.want)
				require.Equal(t, model, gjson.GetBytes(wire, "model").String())
				require.Equal(t, "response.create", gjson.GetBytes(wire, "type").String())
				require.Equal(t, "Caller developer.", gjson.GetBytes(wire, "input.0.content").String())
			})
		}
	}
}

func TestGPT62InstructionsNativeWebSocketPreservesFields(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		for _, mode := range []string{OpenAIWSIngressModeCtxPool, OpenAIWSIngressModePassthrough, OpenAIWSIngressModeHTTPBridge} {
			for _, tc := range gpt62InstructionsCases {
				t.Run(fmt.Sprintf("%s/%s/%s", model, mode, tc.name), func(t *testing.T) {
					capture := &openAIWSCaptureConn{events: [][]byte{[]byte(gpt62InstructionsCompleted)}}
					httpUpstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: " + gpt62InstructionsCompleted + "\n\ndata: [DONE]\n\n"))}}
					svc := wsSSETestService(t, &wsSSETestDialer{conn: capture}, httpUpstream)
					svc.cfg.Gateway.OpenAIWS.ModeRouterV2Enabled = true
					account := wsSSETestAccount()
					account.Extra["openai_oauth_responses_websockets_v2_mode"] = mode
					if mode == OpenAIWSIngressModePassthrough {
						svc.openaiWSPassthroughDialer = &openAIWSCaptureDialer{conn: capture}
					}
					observed := make(chan error, 2)
					done := make(chan error, 1)
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						conn, err := coderws.Accept(w, r, nil)
						if err != nil {
							done <- err
							return
						}
						defer func() { _ = conn.CloseNow() }()
						ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
						defer cancel()
						_, first, err := conn.Read(ctx)
						if err != nil {
							done <- err
							return
						}
						c, _ := gin.CreateTestContext(httptest.NewRecorder())
						c.Request = r.Clone(ctx)
						hooks := &OpenAIWSIngressHooks{AfterTurn: func(_ int, _ *OpenAIForwardResult, err error) { observed <- err }}
						done <- svc.ProxyResponsesWebSocketFromClient(ctx, c, conn, account, "fixture-token", first, hooks)
					}))
					defer server.Close()
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					defer cancel()
					client, _, err := coderws.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
					require.NoError(t, err)
					defer func() { _ = client.CloseNow() }()
					payload := map[string]any{"type": "response.create", "model": model, "input": []any{map[string]any{"role": "developer", "content": "Caller developer."}, map[string]any{"role": "user", "content": "hello"}}}
					if tc.present {
						payload["instructions"] = tc.value
					}
					body, err := json.Marshal(payload)
					require.NoError(t, err)
					require.NoError(t, client.Write(ctx, coderws.MessageText, body))
					_, response, err := client.Read(ctx)
					require.NoError(t, err)
					require.Equal(t, "response.completed", gjson.GetBytes(response, "type").String())
					select {
					case err := <-observed:
						require.NoError(t, err)
					case <-ctx.Done():
						t.Fatal("missing completed WS turn")
					}
					_ = client.CloseNow()
					select {
					case <-done:
					case <-ctx.Done():
						t.Fatal("WS session did not stop")
					}
					var wire []byte
					if mode == OpenAIWSIngressModeHTTPBridge {
						require.Len(t, httpUpstream.bodies, 1)
						wire = httpUpstream.lastBody
					} else {
						require.Empty(t, httpUpstream.requests)
						capture.mu.Lock()
						wire, err = json.Marshal(capture.lastWrite)
						capture.mu.Unlock()
						require.NoError(t, err)
					}
					require.Equal(t, model, gjson.GetBytes(wire, "model").String())
					require.Equal(t, tc.present, gjson.GetBytes(wire, "instructions").Exists())
					if tc.present {
						expected, err := json.Marshal(tc.value)
						require.NoError(t, err)
						require.JSONEq(t, string(expected), gjson.GetBytes(wire, "instructions").Raw)
					}
					requireNoGeneratedCodexTemplate(t, wire)
					require.Equal(t, "Caller developer.", gjson.GetBytes(wire, "input.0.content").String())
					require.Equal(t, "hello", gjson.GetBytes(wire, "input.1.content").String())
				})
			}
		}
	}
}
