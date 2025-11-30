import torch


class PPOBuffer:
    """
    Buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, size, num_envs, device, gamma=0.99, gae_lambda=0.95):
        # Initialize buffer
        self.capacity = size
        # Observation and action buffers: time-major then env-major.
        self.obs_buf = torch.zeros((size, num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, num_envs, *act_dim), dtype=torch.float32, device=device)
        # Scalars per (time,env)
        self.rew_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.term_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.trunc_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.logprob_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr = 0

    def store(self, obs, act, rew, val, term, trunc, logprob):
        """Store a single time-step for all environments at index self.ptr."""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.term_buf[self.ptr] = term
        self.trunc_buf[self.ptr] = trunc
        self.logprob_buf[self.ptr] = logprob
        self.ptr += 1

    def calculate_advantages(self, last_vals, last_terminateds, last_truncateds):
        """Compute GAE advantages and returns.

        Handles terminal/truncated edges using provided "last" flags/values for the step after the buffer.
        Returns:
          - adv_buf: advantage estimates (T,N)
          - ret_buf: adv + value -> target returns
        """
        assert self.ptr == self.capacity, "Buffer not full"
        with torch.no_grad():
            adv_buf = torch.zeros_like(self.rew_buf)
            last_gae = torch.zeros(self.rew_buf.shape[1], device=self.rew_buf.device)
            for t in reversed(range(self.capacity)):
                # next value and masks come from the step after t (use last_* for final step)
                if t == self.capacity - 1:
                    next_vals = last_vals
                    term_mask = 1.0 - last_terminateds
                    trunc_mask = 1.0 - last_truncateds
                else:
                    next_vals = self.val_buf[t + 1]
                    term_mask = 1.0 - self.term_buf[t + 1]
                    trunc_mask = 1.0 - self.trunc_buf[t + 1]

                delta = self.rew_buf[t] + self.gamma * next_vals * term_mask - self.val_buf[t]
                # If next step was terminal/truncated, GAE stops/adjusts via masks.
                last_gae = delta + self.gamma * self.gae_lambda * term_mask * trunc_mask * last_gae
                adv_buf[t] = last_gae

            ret_buf = adv_buf + self.val_buf
            return adv_buf, ret_buf

    def get(self):
        """Retrieve buffers for training and reset write pointer."""
        assert self.ptr == self.capacity
        self.ptr = 0
        return self.obs_buf, self.act_buf, self.logprob_buf
