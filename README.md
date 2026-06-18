# GSM8K-RL

This is part of a series of small projects I'm working on to brush up on RL concepts and get familiar with the modern LLM training stack.

In this project I'll try to implement GRPO (Group Relative Policy Optimization) on GSM8K - grade school math problems. The idea is to fine-tune a small language model using RL, where the reward signal is simply exact match on the final numeric answer.

This is also my first time working with [verifiers](https://github.com/PrimeIntellect-ai/verifiers) and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). I'm really liking the interface they've built - you just define your environment and reward function with verifiers, write a TOML config, and prime-rl handles the rest of the training loop. It abstracts away a lot of the complexity I would have otherwise had to wire up manually with TRL.

I'll be using Qwen3-1.7B - GSM8K is fairly simple so a 1.7B model should be more than enough - W&B for logging, and prime-rl's GRPO trainer.

Before training I'll follow the advice from the verifiers docs: evaluate baseline performance first to make sure the model gets non-zero reward on GSM8K (if it gets 0% after 10+ attempts, the task is too hard and RL won't help). If the baseline is already 80%+, I'll consider filtering to harder examples instead.

I'll start with just a correctness reward and maybe compare it against a run that also includes a formatting reward (encouraging the model to produce its answer in the `####` format) to see if the additional signal helps.
