package repository

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

func TestOpenAI409RedisShadowRecordSize(t *testing.T) {
	cache, _ := newHTTPResponseOwnerTestCache(t)
	ctx := t.Context()
	principal, task := strings.Repeat("a", 64), strings.Repeat("b", 64)
	body := strings.Repeat("x", 256*1024)

	require.NoError(t, cache.PutOpenAIReadableShadow(ctx, principal, task, body, time.Hour))
	require.Error(t, cache.PutOpenAIReadableShadow(ctx, principal, task, body+"x", time.Hour))
	stored, err := cache.GetOpenAIReadableShadow(ctx, principal, task)
	require.NoError(t, err)
	require.Equal(t, body, stored)
}

func TestOpenAI409RedisRetentionLimits(t *testing.T) {
	cache, mr := newHTTPResponseOwnerTestCache(t)
	ctx := t.Context()
	principal := strings.Repeat("a", 64)
	oldest := fmt.Sprintf("%064x", 1)
	last := fmt.Sprintf("%064x", 2)
	overflow := fmt.Sprintf("%064x", 3)
	bodyKey, bodyIndex, err := openAIReadableShadowKeys(principal, oldest)
	require.NoError(t, err)
	epochKey, epochIndex, err := openAIAgentShadowEpochKeys(principal + ":" + oldest)
	require.NoError(t, err)
	toolKey, toolIndex, err := openAIToolProvenanceKeys(principal + ":" + oldest)
	require.NoError(t, err)

	for _, tc := range []struct {
		name, key, index string
		limit            int
		put              func(string, time.Duration) error
	}{
		{"body", bodyKey, bodyIndex, service.OpenAIReadableShadowMaxRecords, func(task string, ttl time.Duration) error {
			return cache.PutOpenAIReadableShadow(ctx, principal, task, "fixture", ttl)
		}},
		{"epoch", epochKey, epochIndex, service.OpenAIRecoveryMetadataMaxRecords, func(task string, ttl time.Duration) error {
			return cache.ReserveOpenAIAgentRecoveryEpoch(ctx, principal+":"+task, "pending:fixture", ttl)
		}},
		{"tool", toolKey, toolIndex, service.OpenAIRecoveryMetadataMaxRecords, func(task string, ttl time.Duration) error {
			return cache.ReserveOpenAIToolProvenance(ctx, principal+":"+task, "pending:fixture", ttl)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.NoError(t, tc.put(oldest, 40*time.Minute))
			require.Equal(t, 40*time.Minute, mr.TTL(tc.key))

			// Seed only the intermediate index members. Both boundary inserts
			// use the normal repository methods and evict a real stored value.
			members := make([]redis.Z, tc.limit-2)
			score := float64(time.Now().Add(50 * time.Minute).UnixMilli())
			for i := range members {
				members[i] = redis.Z{Score: score, Member: fmt.Sprintf("fixture-%d", i)}
			}
			require.NoError(t, cache.rdb.ZAdd(ctx, tc.index, members...).Err())
			require.NoError(t, tc.put(last, time.Hour))
			require.EqualValues(t, tc.limit, cache.rdb.ZCard(ctx, tc.index).Val())
			require.True(t, mr.Exists(tc.key), "the last free slot must not evict a live record")
			require.Equal(t, time.Hour, mr.TTL(tc.index))

			require.NoError(t, tc.put(last, time.Hour))
			require.True(t, mr.Exists(tc.key), "a duplicate must not consume another slot")
			require.NoError(t, tc.put(overflow, time.Hour))
			require.False(t, mr.Exists(tc.key), "overflow must remove the earliest-expiring record")
			require.EqualValues(t, tc.limit, cache.rdb.ZCard(ctx, tc.index).Val())
		})
	}
}
