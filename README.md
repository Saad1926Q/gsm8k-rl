# GSM8K-RL

This is part of a series of small projects I'm working on to brush up on RL concepts and get familiar with the modern LLM training stack.

In this project I'll try to implement GRPO (Group Relative Policy Optimization) on GSM8K - grade school math problems. The idea is to fine-tune a small language model using RL, where the reward signal is simply exact match on the final numeric answer. No LLM judge, no ambiguity.

I plan on using Qwen3-1.5B or SmolLM2-1.7B (small enough to iterate fast on a single GPU), TRL's GRPO trainer, and W&B for logging. The rollout loop samples a group of N completions per prompt, scores them, computes group-normalized advantages, and does a policy gradient update with a KL penalty to keep the model from drifting too far from the reference.
