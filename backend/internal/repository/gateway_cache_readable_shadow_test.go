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
	body := strings.Repeat("x", 1536*1024)

	require.NoError(t, cache.PutOpenAIReadableShadow(ctx, principal, task, body, 30*time.Minute))
	require.Error(t, cache.PutOpenAIReadableShadow(ctx, principal, task, body+"x", 30*time.Minute))
	require.Error(t, cache.PutOpenAIReadableShadow(ctx, principal, task, body, 30*time.Minute+time.Second))
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
			require.NoError(t, tc.put(oldest, 20*time.Minute))
			require.Equal(t, 20*time.Minute, mr.TTL(tc.key))

			// Seed only the intermediate index members. Both boundary inserts
			// use the normal repository methods and evict a real stored value.
			members := make([]redis.Z, tc.limit-2)
			score := float64(time.Now().Add(25 * time.Minute).UnixMilli())
			for i := range members {
				members[i] = redis.Z{Score: score, Member: fmt.Sprintf("fixture-%d", i)}
			}
			require.NoError(t, cache.rdb.ZAdd(ctx, tc.index, members...).Err())
			require.NoError(t, tc.put(last, 30*time.Minute))
			require.EqualValues(t, tc.limit, cache.rdb.ZCard(ctx, tc.index).Val())
			require.True(t, mr.Exists(tc.key), "the last free slot must not evict a live record")
			require.Equal(t, 30*time.Minute, mr.TTL(tc.index))

			require.NoError(t, tc.put(last, 30*time.Minute))
			require.True(t, mr.Exists(tc.key), "a duplicate must not consume another slot")
			require.NoError(t, tc.put(overflow, 30*time.Minute))
			require.False(t, mr.Exists(tc.key), "overflow must remove the earliest-expiring record")
			require.EqualValues(t, tc.limit, cache.rdb.ZCard(ctx, tc.index).Val())
		})
	}
}

func TestOpenAI409RedisLegacyIndexShrinksInBatches(t *testing.T) {
	cache, mr := newHTTPResponseOwnerTestCache(t)
	ctx := t.Context()
	principal := strings.Repeat("a", 64)
	task := strings.Repeat("b", 64)
	key, index, err := openAIReadableShadowKeys(principal, task)
	require.NoError(t, err)
	limit := service.OpenAIReadableShadowMaxRecords
	legacy := make([]redis.Z, limit+80)
	score := float64(time.Now().Add(time.Hour).UnixMilli())
	for i := range legacy {
		legacy[i] = redis.Z{Score: score, Member: fmt.Sprintf("legacy-%04d", i)}
	}
	oldest := legacy[0].Member.(string)
	last := legacy[len(legacy)-1].Member.(string)
	require.NoError(t, cache.rdb.Set(ctx, oldest, "oldest", time.Hour).Err())
	require.NoError(t, cache.rdb.Set(ctx, last, "retained", time.Hour).Err())
	require.NoError(t, cache.rdb.ZAdd(ctx, index, legacy...).Err())
	require.NoError(t, cache.rdb.PExpire(ctx, index, time.Hour).Err())

	require.NoError(t, cache.PutOpenAIReadableShadow(ctx, principal, task, "new", 30*time.Minute))
	require.False(t, mr.Exists(oldest))
	require.False(t, mr.Exists(key), "do not add bytes to a legacy index still over budget")
	require.EqualValues(t, limit+16, cache.rdb.ZCard(ctx, index).Val())

	require.NoError(t, cache.PutOpenAIReadableShadow(ctx, principal, task, "new", 30*time.Minute))
	require.True(t, mr.Exists(key))
	require.Equal(t, 30*time.Minute, mr.TTL(key))
	require.EqualValues(t, limit, cache.rdb.ZCard(ctx, index).Val())
	require.Equal(t, time.Hour, mr.TTL(last))
}
