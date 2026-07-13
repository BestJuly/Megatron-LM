# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed gradient check for the MDP CP-local vision bridge.

Run with two ranks, for example:

```
torchrun --nproc_per_node 2 tests/unit_tests/multimodal/test_mdp_gradient_flow.py
```

The test intentionally keeps the model tiny. Each CP rank owns one image shard,
all-gathers vision rows through ``gather_to_inner_dp_zero``, selects the rows
needed by its CP-local text shard, scatters them into the local sequence, and
backpropagates. The expected gradients are hand-computed per rank so the test
checks both gradient flow and rank-specific row ownership.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist

from examples.multimodal_dev.modality_bridge import (
    cp_local_image_positions_and_row_ids_from_cpu_metadata,
    gather_to_inner_dp_zero,
    reorder_gathered_embeddings,
    scatter_vision_rows_at_positions,
    select_vision_rows_for_cp_rank,
)


def _distributed_ready() -> bool:
    return (
        dist.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) == 2
    )


class TestMdpCpLocalVisionGradient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not _distributed_ready():
            raise unittest.SkipTest("requires torchrun with WORLD_SIZE=2")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            # Bind each process before ProcessGroupNCCL initialization.
            # Initializing first can let both local ranks select the default
            # device and intermittently stall the first collective.
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend=backend)

    @classmethod
    def tearDownClass(cls):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def test_gradient_reaches_rank_owned_vision_rows(self):
        rank = dist.get_rank()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if dist.get_backend() != "nccl":
            raise unittest.SkipTest(
                "gather_to_inner_dp_zero uses Megatron's sequence-parallel "
                "reduce-scatter in backward; this gradient contract is "
                "validated on NCCL."
            )

        # Rank 0 owns image 0 rows, rank 1 owns image 1 rows.
        rank_assignment = {0: [(0, 0)], 1: [(0, 1)]}
        global_per_image_row_counts = [2, 2]

        if rank == 0:
            local_pixels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
            expected_local_embedding_grad = torch.tensor(
                [[10.0, 100.0], [30.0, 300.0]], device=device
            )
            expected_weight_grad = torch.tensor(
                [[10.0, 30.0, 0.0], [100.0, 300.0, 0.0]], device=device
            )
        else:
            local_pixels = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], device=device)
            expected_local_embedding_grad = torch.tensor(
                [[40.0, 400.0], [20.0, 200.0]], device=device
            )
            expected_weight_grad = torch.tensor(
                [[20.0, 20.0, 40.0], [200.0, 200.0, 400.0]], device=device
            )

        encoder = torch.nn.Linear(3, 2, bias=False, device=device)
        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device))

        local_embeddings = encoder(local_pixels)
        local_embeddings.retain_grad()

        gathered = gather_to_inner_dp_zero(
            local_embeddings=local_embeddings,
            rank_assignment=rank_assignment,
            encoder_dp_group=dist.group.WORLD,
            global_per_image_row_counts=global_per_image_row_counts,
        )
        canonical = reorder_gathered_embeddings(
            gathered_embeddings=gathered,
            local_per_image_row_counts=None,
            rank_assignment=rank_assignment,
            group=dist.group.WORLD,
            global_per_image_row_counts=global_per_image_row_counts,
        )

        # Full image-token order is:
        #   row 0 -> position 1, row 1 -> position 2,
        #   row 2 -> position 5, row 3 -> position 6.
        # With CP=2 zigzag sharding:
        #   rank 0 consumes rows [0, 3], rank 1 consumes rows [1, 2].
        image_positions_cp, cp_local_row_ids, _ = (
            cp_local_image_positions_and_row_ids_from_cpu_metadata(
                image_positions=[1, 2, 5, 6], input_shape=(1, 8), cp_size=2, cp_rank=rank
            )
        )
        expected_row_ids = (
            torch.tensor([0, 3], dtype=torch.int64)
            if rank == 0
            else torch.tensor([1, 2], dtype=torch.int64)
        )
        self.assertTrue(torch.equal(cp_local_row_ids.cpu(), expected_row_ids))

        selected = select_vision_rows_for_cp_rank(canonical, cp_local_row_ids)
        text_embeddings = torch.zeros(4, 1, 2, device=device)
        decoder_input = scatter_vision_rows_at_positions(
            text_embeddings, selected, image_positions_cp
        )

        weights = torch.zeros_like(decoder_input)
        if rank == 0:
            weights[1, 0] = torch.tensor([10.0, 100.0], device=device)
            weights[2, 0] = torch.tensor([20.0, 200.0], device=device)
        else:
            weights[0, 0] = torch.tensor([30.0, 300.0], device=device)
            weights[3, 0] = torch.tensor([40.0, 400.0], device=device)

        loss = (decoder_input * weights).sum()
        loss.backward()

        self.assertTrue(
            torch.allclose(
                local_embeddings.grad, expected_local_embedding_grad, atol=0.0, rtol=0.0
            ),
            msg=(
                f"rank={rank} local embedding grad "
                f"{local_embeddings.grad} != {expected_local_embedding_grad}"
            ),
        )
        self.assertTrue(
            torch.allclose(encoder.weight.grad, expected_weight_grad, atol=0.0, rtol=0.0),
            msg=(
                f"rank={rank} encoder weight grad "
                f"{encoder.weight.grad} != {expected_weight_grad}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
