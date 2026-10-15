# sci-nodes

`sci-nodes` provides scipy-dependent nodes for **nonoha** — a lightweight
node-based data processing framework. It contains the streaming filter and
resampling nodes that were split out of [nonoha-core](https://github.com/viestyle-dev/nonoha-core)
when the core went scipy-free, so that the core stays NumPy-only while
scipy-backed DSP remains available where scipy is installed.

## Installation

Install from the repository:

```sh
pip install git+https://github.com/viestyle-dev/sci-nodes.git
```

Requires Python 3.11+. Runtime dependencies are NumPy and SciPy
(`scipy.signal`, public API only).

## Nodes

| Registered name | Class | Description |
| --- | --- | --- |
| `BandPassFilter` | `BandPassFilterNode` | Butterworth band-pass filter |
| `HighPassFilter` | `HighPassFilterNode` | Butterworth high-pass filter |
| `LowPassFilter` | `LowPassFilterNode` | Butterworth low-pass filter |
| `NotchFilter` | `NotchFilterNode` | IIR notch filter (`iirnotch`) |
| `SmaFilter` | `SmaFilterNode` | Simple moving average (FIR) |
| `ResamplePoly` | `ResamplePolyNode` | Polyphase resampling, streaming-safe output matching `scipy.signal.resample_poly` |

All filter nodes are built for continuous streaming data: filter state
(`zi`) is carried across batches, so chunked input produces the same
output as one continuous array. `ResamplePoly` uses symmetric overlap
buffering to produce output identical to scipy's offline
`resample_poly`.

## Usage

Shared registration contract — works in every nonoha runtime:

```python
from nonoha_core.node_registry import NodeRegistry
import sci_nodes

registry = NodeRegistry()
sci_nodes.register_nodes(registry)
```

The node classes and registered type names are kept in lockstep with the
nonoha engine, so graphs authored there load unchanged in other runtimes
that register these nodes.

`nonoha-core` is provided by the host runtime, so it is a dev dependency
only (see `pyproject.toml`).

## Testing

```sh
uv run pytest
```

The dev dependency group installs `pytest` and `nonoha-core`.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Pull requests require signing the [nonoha CLA](CLA.md) (a one-time
comment on your first PR, shared across the nonoha repositories).

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
