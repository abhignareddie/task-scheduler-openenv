---
title: Task Scheduler OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Task Scheduler OpenEnv

A real-world task scheduling environment where an LLM agent learns to prioritize and schedule tasks to maximize reward based on deadlines and priorities.

## Environment Description
The agent selects which task to execute next from a list. Each task has a deadline, priority, and duration. The agent must order tasks to maximize cumulative reward.

## Action Space
Integer index (0 to N-1) — selects which task to execute next.

## Observation Space
- `tasks`: list of tasks with id, deadline, priority, duration
- `current_time`: elapsed time

## Tasks
- **Easy**: 3 tasks, generous deadlines
- **Medium**: 3 tasks, tighter deadlines, higher priority variation
- **Hard**: 4 tasks, strict deadlines, requires optimal ordering

## Reward Function
- Positive reward for completing tasks before deadline
- Partial reward for late completion based on priority

## Setup
```bash