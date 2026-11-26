package service

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestOpenAINativeWSContextRejectionPreservesSchedulerIdentity(t *testing.T) {
	for _, mode := range []string{OpenAIWSIngressModeCtxPool, OpenAIWSIngressModePassthrough} {
		for _, event := range []string{"error", "response.failed"} {
			t.Run(mode+"/"+event, func(t *testing.T) {
				message := []byte(`{"type":"error","error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"too long"}}`)
				if event == "response.failed" {
					message = []byte(`{"type":"response.failed","response":{"id":"context-rejected","status":"failed","error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"too long"},"usage":{"input_tokens":7,"output_tokens":0}}}`)
				}
				upstream := &openAIWSCaptureConn{events: [][]byte{message}}
				svc := wsSSETestService(t, &wsSSETestDialer{conn: upstream}, &httpUpstreamRecorder{})
				svc.cfg.Gateway.OpenAIWS.ModeRouterV2Enabled = true
				account := wsSSETestAccount()
				account.Extra["openai_oauth_responses_websockets_v2_mode"] = mode
				if mode == OpenAIWSIngressModePassthrough {
					svc.openaiWSPassthroughDialer = &openAIWSCaptureDialer{conn: upstream}
				}
				type turnObservation struct {
					result               *OpenAIForwardResult
					err                  error
					schedulerInitialized bool
				}
				observed := make(chan turnObservation, 2)
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
					hooks := &OpenAIWSIngressHooks{AfterTurn: func(_ int, result *OpenAIForwardResult, turnErr error) {
						svc.ReportOpenAIAccountScheduleResult(account, "gpt-6-astra", result.SucceededForScheduling(), nil, turnErr)
						observed <- turnObservation{result, turnErr, svc.openaiScheduler != nil}
					}}
					err = svc.ProxyResponsesWebSocketFromClient(ctx, c, conn, account, "fixture-token", first, hooks)
					var closeErr *OpenAIWSClientCloseError
					if errors.As(err, &closeErr) {
						_ = conn.Close(closeErr.StatusCode(), closeErr.Reason())
					}
					done <- err
				}))
				defer server.Close()
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				client, _, err := coderws.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
				require.NoError(t, err)
				defer func() { _ = client.CloseNow() }()
				require.NoError(t, client.Write(ctx, coderws.MessageText, []byte(`{"type":"response.create","model":"gpt-6-astra","input":"hi"}`)))
				_, response, err := client.Read(ctx)
				require.NoError(t, err)
				require.JSONEq(t, string(message), string(response))
				select {
				case turn := <-observed:
					require.ErrorIs(t, turn.err, errGatewayContextRejection)
					require.False(t, turn.schedulerInitialized)
					if turn.result != nil {
						require.False(t, turn.result.SucceededForScheduling())
					}
					if event == "response.failed" {
						require.NotNil(t, turn.result)
						require.Equal(t, 7, turn.result.Usage.InputTokens)
						require.True(t, turn.result.ObservedUsageOnError)
					}
				case <-ctx.Done():
					t.Fatal("missing per-turn context rejection")
				}
				_ = client.CloseNow()
				select {
				case <-done:
				case <-ctx.Done():
					t.Fatal("native websocket did not shut down")
				}
			})
		}
	}
}

func TestOpenAINativeWSContextRejectionDoesNotLeakAcrossTurns(t *testing.T) {
	for _, mode := range []string{OpenAIWSIngressModeCtxPool, OpenAIWSIngressModePassthrough} {
		for _, errorPrefix := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/error_prefix=%t", mode, errorPrefix), func(t *testing.T) {
				upstream := newStagedPassthroughConn()
				svc := wsSSETestService(t, &wsSSETestDialer{conn: &ingressDrainTestConn{upstream}}, &httpUpstreamRecorder{})
				svc.cfg.Gateway.OpenAIWS.ModeRouterV2Enabled = true
				account := wsSSETestAccount()
				account.Extra["openai_oauth_responses_websockets_v2_mode"] = mode
				if mode == OpenAIWSIngressModePassthrough {
					svc.openaiWSPassthroughDialer = &stagedPassthroughDialer{conn: upstream}
				}
				type observation struct {
					result *OpenAIForwardResult
					err    error
				}
				observed := make(chan observation, 4)
				done := make(chan error, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					conn, err := coderws.Accept(w, r, nil)
					if err != nil {
						done <- err
						return
					}
					defer func() { _ = conn.CloseNow() }()
					ctx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
					defer cancel()
					_, first, err := conn.Read(ctx)
					if err != nil {
						done <- err
						return
					}
					c, _ := gin.CreateTestContext(httptest.NewRecorder())
					c.Request = r.Clone(ctx)
					hooks := &OpenAIWSIngressHooks{AfterTurn: func(_ int, result *OpenAIForwardResult, err error) { observed <- observation{result, err} }}
					done <- svc.ProxyResponsesWebSocketFromClient(ctx, c, conn, account, "fixture-token", first, hooks)
				}))
				defer server.Close()
				ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
				defer cancel()
				client, _, err := coderws.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
				require.NoError(t, err)
				defer func() { _ = client.CloseNow() }()
				for turn, terminal := range []string{
					`{"type":"response.failed","response":{"id":"context-rejected","status":"failed","error":{"code":"context_length_exceeded","message":"too long"},"usage":{"input_tokens":7,"output_tokens":0}}}`,
					`{"type":"response.failed","response":{"id":"provider-failed","status":"failed","error":{"code":"unknown_provider_error","message":"provider failure"},"usage":{"input_tokens":3,"output_tokens":0}}}`,
					`{"type":"response.completed","response":{"id":"completed","status":"completed","usage":{"input_tokens":2,"output_tokens":1}}}`,
				} {
					require.NoError(t, client.Write(ctx, coderws.MessageText, []byte(`{"type":"response.create","model":"gpt-6-astra","input":"short request"}`)))
					select {
					case <-upstream.writes:
					case <-ctx.Done():
						t.Fatal("missing upstream turn")
					}
					if turn == 0 && errorPrefix {
						upstream.Send(`{"type":"error","error":{"code":"context_length_exceeded","message":"too long"}}`)
						_, _, err := client.Read(ctx)
						require.NoError(t, err)
					}
					upstream.Send(terminal)
					_, payload, err := client.Read(ctx)
					require.NoError(t, err)
					require.JSONEq(t, terminal, string(payload))
					select {
					case outcome := <-observed:
						require.NotNil(t, outcome.result)
						if turn == 0 {
							require.ErrorIs(t, outcome.err, errGatewayContextRejection)
							require.Equal(t, 7, outcome.result.Usage.InputTokens)
							require.True(t, outcome.result.ObservedUsageOnError)
						} else {
							require.NoError(t, outcome.err)
						}
						require.Equal(t, turn == 2, outcome.result.SucceededForScheduling())
					case <-ctx.Done():
						t.Fatal("missing turn callback")
					}
				}
				_ = client.CloseNow()
				select {
				case <-done:
				case <-ctx.Done():
					t.Fatal("native websocket did not shut down")
				}
			})
		}
	}
}
