# Agent SNR

An open-source, lightweight, framework-agnostic **Agent Signal Firewall** in Python.

`agent-snr` monitors and filters the context window of autonomous agents in real time, maximizing Signal-to-Noise Ratio (SNR) through intelligent compression, compaction, entropy tracking, and circuit breaking.

## Features

- **Framework Agnostic**: Plugs into LangGraph, Autogen, CrewAI, or vanilla agent loops.
- **Zero Model Dependencies by Default**: Built-in deterministic 384-d feature-hashing embedder (`HashNgramEmbedder`) runs out-of-the-box without downloading gigabytes of model weights.
- **Fast Embedding Support**: Optional fast on-device inference via `FastEmbedAdapter` (`BAAI/bge-small-en-v1.5`).
- **Signal Accounting**: Tracks raw tokens, compacted tokens, hops skipped, semantic drift, and tokens saved across agent sessions.

## Installation

```bash
# Core installation
pip install agent-snr

# With fastembed support
pip install "agent-snr[embeddings]"

# With LangGraph integration
pip install "agent-snr[langgraph]"
```

## Quick Start

```python
from agent_snr import SignalFrame, FrameRole, TraceLedger, HashNgramEmbedder

embedder = HashNgramEmbedder()
ledger = TraceLedger(session_id="session-001")

frame = SignalFrame(
    step_id=1,
    node_id="planner",
    role=FrameRole.USER,
    content="Find the latest financial reports",
    raw_tokens=32,
    embedding=embedder.embed("Find the latest financial reports"),
)

ledger.add_frame(frame)
```

## License

Apache-2.0
