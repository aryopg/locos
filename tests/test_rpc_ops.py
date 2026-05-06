class TestRemapHeadsForTP:
    def test_rank0_gets_first_shard(self):
        from locos_eval.rpc_ops import _remap_heads_for_tp

        # 32 global heads, TP=4 → 8 heads per shard
        # Rank 0 owns global heads 0-7
        heads = [(0, 0), (0, 7), (0, 8), (1, 15), (1, 31)]
        result = _remap_heads_for_tp(heads, num_heads_per_shard=8, tp_rank=0)
        assert result == [(0, 0), (0, 7)]

    def test_rank2_gets_middle_shard(self):
        from locos_eval.rpc_ops import _remap_heads_for_tp

        # Rank 2 owns global heads 16-23
        heads = [(0, 0), (0, 16), (0, 23), (0, 24), (1, 20)]
        result = _remap_heads_for_tp(heads, num_heads_per_shard=8, tp_rank=2)
        # 16→0, 23→7, 20→4
        assert result == [(0, 0), (0, 7), (1, 4)]

    def test_no_heads_on_rank(self):
        from locos_eval.rpc_ops import _remap_heads_for_tp

        # All heads on rank 0, rank 3 gets nothing
        heads = [(0, 0), (0, 1), (1, 2)]
        result = _remap_heads_for_tp(heads, num_heads_per_shard=8, tp_rank=3)
        assert result == []

    def test_tp1_returns_all(self):
        from locos_eval.rpc_ops import _remap_heads_for_tp

        # TP=1 → 32 heads per shard, rank 0 owns all
        heads = [(0, 0), (0, 31), (1, 15)]
        result = _remap_heads_for_tp(heads, num_heads_per_shard=32, tp_rank=0)
        assert result == heads
