import chex
import flax.linen as nn


class BatchNorm(nn.BatchNorm):

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        mutable = self.is_mutable_collection('batch_stats')
        return super().__call__(x, use_running_average=not mutable)
