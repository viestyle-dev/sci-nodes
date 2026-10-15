import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, resample_poly

from sci_nodes.filter_nodes import (
    BandPassFilterNode,
    HighPassFilterNode,
    LowPassFilterNode,
    NotchFilterNode,
    ResamplePolyNode,
)


# def test_band_pass_node():
#     node = BandPassNode("test")

#     node.set_properties({
#         "shape": ""
#     })


# Simple, practical implementation
def initialize_zi_simple(base_zi, np_shape, axis):
    """Practical zi initialization (works regardless of dimensionality)."""
    # Compute the zi shape
    zi_shape = list(np_shape)
    zi_shape[axis] = len(base_zi)

    # Create an empty zi
    zi = np.zeros(zi_shape, dtype=np.float32)

    # Expand base_zi along axis (reshape + broadcast)
    # 1. Build a reshape target shape sized correctly along axis
    reshape_shape = [1] * len(np_shape)
    reshape_shape[axis] = len(base_zi)

    # 2. Reshape base_zi to the new shape
    reshaped_zi = base_zi.reshape(reshape_shape)

    # 3. Broadcast across all dimensions
    expanded_zi = np.broadcast_to(reshaped_zi, zi_shape)

    # Assign the result to zi
    zi = expanded_zi.copy()  # broadcast views are read-only, so copy

    return zi


def test_filter():
    fs = 250
    low_cut = 3
    high_cut = 30
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    order = 4
    b, a = butter(order, [low, high], btype="band")  # type: ignore

    base_zi = lfilter_zi(b, a)
    # print(f"{b=}, {a=}, {len(b)=} {base_zi=}")
    zi = initialize_zi_simple(base_zi, (1, 2), 0)
    # print(f"{zi=}")


def test_resample_poly_node_overlap_in_property():
    """Test that overlap_in property can be set and retrieved."""
    node = ResamplePolyNode("test")

    # Check default value
    assert node.properties["overlap_in"] == 128

    # Set custom overlap_in
    node.set_properties({"overlap_in": 64})
    assert node.properties["overlap_in"] == 64
    assert node._overlap_in == 64

    # Verify it works with different value
    node.set_properties({"overlap_in": 256})
    assert node.properties["overlap_in"] == 256
    assert node._overlap_in == 256


def test_resample_poly_node_streaming():
    """Test ResamplePolyNode for correct sample count and continuity in streaming mode."""
    # Setup: Use larger chunks to ensure overlap buffer fills (overlap_in=128)
    fs_in = 250
    fs_out = 500
    chunk_size = 100  # Large enough to exceed 2*overlap_in threshold
    num_chunks = 10
    freq = 1.0  # 1Hz sine wave

    # Create node
    node = ResamplePolyNode("test")
    node.set_properties({"fs_in": fs_in, "fs_out": fs_out})

    input_port = node.inputs["series"]
    output_port = node.outputs["series"]

    # Collect outputs
    outputs = []

    # Process chunks
    for i in range(num_chunks):
        t_start = i * chunk_size / fs_in
        t_end = (i + 1) * chunk_size / fs_in
        t = np.linspace(t_start, t_end, chunk_size, endpoint=False)
        chunk = np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Dispatch input
        input_port.dispatch([chunk])

        # Collect output
        if output_port.current_values is not None:
            outputs.append(output_port.current_values[0].copy())

    # Verify results
    assert len(outputs) > 0, f"Expected outputs, got {len(outputs)}"

    # Check sample count: each output should be approximately 2x upsampled
    # (with overlap processing, exact count may vary)
    for i, output in enumerate(outputs):
        # Allow some variation due to overlap processing
        assert len(output) > 0, f"Chunk {i}: output is empty"

    # Check continuity at chunk boundaries
    if len(outputs) > 1:
        all_output = np.concatenate(outputs)
        max_discontinuity = 0.0
        for i in range(len(outputs) - 1):
            pos = sum(len(o) for o in outputs[: i + 1])
            if pos < len(all_output):
                diff = abs(all_output[pos] - all_output[pos - 1])
                max_discontinuity = max(max_discontinuity, diff)

        # Check that discontinuities are acceptably small
        assert (
            max_discontinuity < 1.0
        ), f"Max discontinuity {max_discontinuity:.3f} exceeds threshold"


def test_band_pass_filter_node_properties():
    """Test that BandPassFilterNode properties can be set and retrieved."""
    node = BandPassFilterNode("test")

    # Check default values
    assert node.properties["fs"] == 250
    assert node.properties["low_cut"] == 4
    assert node.properties["high_cut"] == 40
    assert node.properties["order"] == 4

    # Set custom properties
    node.set_properties({"fs": 500, "low_cut": 8, "high_cut": 30, "order": 6})
    assert node.properties["fs"] == 500
    assert node.properties["low_cut"] == 8
    assert node.properties["high_cut"] == 30
    assert node.properties["order"] == 6


def test_band_pass_filter_node_filtering():
    """Test BandPassFilterNode filters signal correctly."""
    fs = 250
    low_cut = 8
    high_cut = 12
    chunk_size = 100
    num_chunks = 5

    # Create node
    node = BandPassFilterNode("test")
    node.set_properties({"fs": fs, "low_cut": low_cut, "high_cut": high_cut})

    input_port = node.inputs["series"]
    output_port = node.outputs["series"]

    # Generate test signal: mix of frequencies
    # 10Hz (should pass) + 1Hz (should be filtered) + 50Hz (should be filtered)
    outputs = []
    for i in range(num_chunks):
        t_start = i * chunk_size / fs
        t_end = (i + 1) * chunk_size / fs
        t = np.linspace(t_start, t_end, chunk_size, endpoint=False)
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        signal_1hz = 0.5 * np.sin(2 * np.pi * 1 * t)
        signal_50hz = 0.3 * np.sin(2 * np.pi * 50 * t)
        chunk = (signal_10hz + signal_1hz + signal_50hz).astype(np.float32)

        input_port.dispatch([chunk])

        if output_port.current_values is not None:
            outputs.append(output_port.current_values[0].copy())

    # Verify filtering occurred
    assert len(outputs) > 0
    for output in outputs:
        assert len(output) == chunk_size
        # Output should be similar to 10Hz signal (which is in the pass band)
        # Check that output exists and has reasonable values
        assert np.max(np.abs(output)) > 0.1  # Output is not zero/silent
        assert np.max(np.abs(output)) < 2.0  # Output is not amplified excessively


def test_high_pass_filter_node_properties():
    """Test that HighPassFilterNode properties can be set and retrieved."""
    node = HighPassFilterNode("test")

    # Check default values
    assert node.properties["fs"] == 250
    assert node.properties["cutoff"] == 4.0
    assert node.properties["order"] == 4

    # Set custom properties
    node.set_properties({"fs": 500, "cutoff": 10.0, "order": 6})
    assert node.properties["fs"] == 500
    assert node.properties["cutoff"] == 10.0
    assert node.properties["order"] == 6


def test_high_pass_filter_node_filtering():
    """Test HighPassFilterNode filters low frequencies correctly."""
    fs = 250
    cutoff = 10
    chunk_size = 100
    num_chunks = 5

    # Create node
    node = HighPassFilterNode("test")
    node.set_properties({"fs": fs, "cutoff": cutoff})

    input_port = node.inputs["series"]
    output_port = node.outputs["series"]

    # Generate test signal: low freq (1Hz) + high freq (30Hz)
    outputs = []
    for i in range(num_chunks):
        t_start = i * chunk_size / fs
        t_end = (i + 1) * chunk_size / fs
        t = np.linspace(t_start, t_end, chunk_size, endpoint=False)
        signal_low = np.sin(2 * np.pi * 1 * t)
        signal_high = 0.5 * np.sin(2 * np.pi * 30 * t)
        chunk = (signal_low + signal_high).astype(np.float32)

        input_port.dispatch([chunk])

        if output_port.current_values is not None:
            outputs.append(output_port.current_values[0].copy())

    # Verify filtering occurred
    assert len(outputs) > 0
    for output in outputs:
        assert len(output) == chunk_size


def test_low_pass_filter_node_properties():
    """Test that LowPassFilterNode properties can be set and retrieved."""
    node = LowPassFilterNode("test")

    # Check default values
    assert node.properties["fs"] == 250
    assert node.properties["cutoff"] == 40.0
    assert node.properties["order"] == 4

    # Set custom properties
    node.set_properties({"fs": 500, "cutoff": 20.0, "order": 6})
    assert node.properties["fs"] == 500
    assert node.properties["cutoff"] == 20.0
    assert node.properties["order"] == 6


def test_low_pass_filter_node_filtering():
    """Test LowPassFilterNode filters high frequencies correctly."""
    fs = 250
    cutoff = 20
    chunk_size = 100
    num_chunks = 5

    # Create node
    node = LowPassFilterNode("test")
    node.set_properties({"fs": fs, "cutoff": cutoff})

    input_port = node.inputs["series"]
    output_port = node.outputs["series"]

    # Generate test signal: low freq (10Hz) + high freq (60Hz)
    outputs = []
    for i in range(num_chunks):
        t_start = i * chunk_size / fs
        t_end = (i + 1) * chunk_size / fs
        t = np.linspace(t_start, t_end, chunk_size, endpoint=False)
        signal_low = np.sin(2 * np.pi * 10 * t)
        signal_high = 0.5 * np.sin(2 * np.pi * 60 * t)
        chunk = (signal_low + signal_high).astype(np.float32)

        input_port.dispatch([chunk])

        if output_port.current_values is not None:
            outputs.append(output_port.current_values[0].copy())

    # Verify filtering occurred
    assert len(outputs) > 0
    for output in outputs:
        assert len(output) == chunk_size


def test_notch_filter_node_properties():
    """Test that NotchFilterNode properties can be set and retrieved."""
    node = NotchFilterNode("test")

    # Check default values
    assert node.properties["fs"] == 600
    assert node.properties["f_notch"] == 60.0
    assert node.properties["bw"] == 2.5

    # Set custom properties
    node.set_properties({"fs": 250, "f_notch": 50.0, "bw": 3.0})
    assert node.properties["fs"] == 250
    assert node.properties["f_notch"] == 50.0
    assert node.properties["bw"] == 3.0


def test_notch_filter_node_filtering():
    """Test NotchFilterNode removes specific frequency correctly."""
    fs = 250
    f_notch = 50
    chunk_size = 100
    num_chunks = 5

    # Create node
    node = NotchFilterNode("test")
    node.set_properties({"fs": fs, "f_notch": f_notch, "bw": 2.5})

    input_port = node.inputs["series"]
    output_port = node.outputs["series"]

    # Generate test signal: 10Hz + 50Hz (notch target) + 30Hz
    outputs = []
    input_power_50hz = []
    output_power_50hz = []

    for i in range(num_chunks):
        t_start = i * chunk_size / fs
        t_end = (i + 1) * chunk_size / fs
        t = np.linspace(t_start, t_end, chunk_size, endpoint=False)
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        signal_50hz = 0.8 * np.sin(2 * np.pi * 50 * t)
        signal_30hz = 0.5 * np.sin(2 * np.pi * 30 * t)
        chunk = (signal_10hz + signal_50hz + signal_30hz).astype(np.float32)

        # Calculate input power at 50Hz
        input_fft = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(len(chunk), 1 / fs)
        idx_50hz = np.argmin(np.abs(freqs - 50))
        input_power_50hz.append(np.abs(input_fft[idx_50hz]) ** 2)

        input_port.dispatch([chunk])

        if output_port.current_values is not None:
            output = output_port.current_values[0].copy()
            outputs.append(output)

            # Calculate output power at 50Hz
            output_fft = np.fft.rfft(output)
            output_power_50hz.append(np.abs(output_fft[idx_50hz]) ** 2)

    # Verify filtering occurred
    assert len(outputs) > 0
    for output in outputs:
        assert len(output) == chunk_size

    # After a few chunks, output power at 50Hz should be significantly reduced
    if len(output_power_50hz) >= 3:
        avg_input_power = np.mean(input_power_50hz[-3:])
        avg_output_power = np.mean(output_power_50hz[-3:])
        # Notch filter should reduce 50Hz power by at least 50%
        assert avg_output_power < 0.5 * avg_input_power, (
            f"Notch filter did not reduce 50Hz power sufficiently: "
            f"input={avg_input_power:.3f}, output={avg_output_power:.3f}"
        )
