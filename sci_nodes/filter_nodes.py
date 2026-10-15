# Filter nodes using SciPy library

import traceback
import logging

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, firwin, resample_poly, upfirdn, iirnotch

from nonoha_core.node import (
    BaseNode,
    NodeDoc,
    PropertyDataType,
    replace_node_input_port,
    replace_node_output_port,
)
from nonoha_core.port import (
    PortValue,
    StaticTypeInputPort,
    StaticTypeOutputPort,
)
from nonoha_core.data_type import DataType, PortValueType, name_to_port_value_type

logger = logging.getLogger(__name__)


class BandPassFilterNode(BaseNode):
    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Butterworth band-pass filter for continuous streaming data",
            category="Filter",
            inputs={"series": ""},
            outputs={"series": ""},
        )

    PROPERTIES = {
        "shape": {"description": "Input shape suffix", "data_type": "str", "default_value": ""},
        "fs": {"description": "Sample rate (Hz)", "data_type": "u32", "default_value": 250},
        "low_cut": {"description": "Low cutoff frequency (Hz)", "data_type": "f64", "default_value": 4.0},
        "high_cut": {"description": "High cutoff frequency (Hz)", "data_type": "f64", "default_value": 40.0},
        "order": {"description": "Filter order", "data_type": "u32", "default_value": 4},
    }

    def __init__(self, name: str):
        super().__init__(name, "BandPassFilter")

        in_type = PortValueType(DataType.Float32)
        low_cut = 4
        high_cut = 40
        fs = 250
        order = 4
        b, a, zi = self._update_filter(fs, low_cut, high_cut, order, in_type.shape)

        input_series = StaticTypeInputPort("series", [in_type])
        output_series = StaticTypeOutputPort("series", [in_type])

        input_series.subscribe(id(self), self._on_series)

        self._fs = fs
        self._low_cut = low_cut
        self._high_cut = high_cut
        self._order = order
        self._b = b
        self._a = a
        self._zi = zi
        self._in_type = in_type

        self._input_series = input_series
        self._output_series = output_series

        self.inputs[input_series.name] = input_series
        self.outputs[output_series.name] = output_series

    def _update_filter(
        self,
        fs: int,
        low_cut: float,
        high_cut: float,
        order: int,
        input_shape: tuple[int, ...],
    ):
        nyquist = 0.5 * fs
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = butter(order, [low, high], btype="band")  # type: ignore
        base_zi = lfilter_zi(b, a)

        if len(input_shape) == 0:
            return b, a, base_zi

        # Fixed to axis=0 (filtering along the time axis)
        axis = 0

        # Convert the nonoha port shape notation to the NumPy form.
        # e.g. (3,) in port notation is actually an (n, 3) NumPy array;
        # the first (sample count) dimension is provisionally set to 1.
        np_shape = (1,) + input_shape

        # Compute the final shape of the zi array
        zi_shape = list(np_shape)
        zi_shape[axis] = len(base_zi)  # along axis: filter length - 1

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

        return b, a, zi

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract and validate properties
        low_cut = self._low_cut
        if "low_cut" in properties:
            low_cut = float(properties["low_cut"])

        high_cut = self._high_cut
        if "high_cut" in properties:
            high_cut = float(properties["high_cut"])

        fs = self._fs
        if "fs" in properties:
            fs = int(properties["fs"])

        order = self._order
        if "order" in properties:
            order = int(properties["order"])

        in_type = self._in_type
        if "shape" in properties:
            shape_str = properties["shape"]
            in_type = name_to_port_value_type(f"f32{shape_str}")

        # 2. Create new ports if necessary
        new_input_series = self._input_series
        new_output_series = self._output_series
        if in_type != self._in_type:
            new_input_series = StaticTypeInputPort("series", [in_type])
            new_output_series = StaticTypeOutputPort("series", [in_type])

        # 3. Update filter coefficients if necessary
        if (
            low_cut != self._low_cut
            or high_cut != self._high_cut
            or fs != self._fs
            or order != self._order
            or in_type != self._in_type
        ):
            b, a, zi = self._update_filter(fs, low_cut, high_cut, order, in_type.shape)
            self._b = b
            self._a = a
            self._zi = zi

        # 4. Replace ports if changed
        if new_input_series != self._input_series:
            replace_node_input_port(
                self, self._input_series, new_input_series, self._on_series
            )
            self._input_series = new_input_series

        if new_output_series != self._output_series:
            replace_node_output_port(self, self._output_series, new_output_series)
            self._output_series = new_output_series

        # 5. Update instance variables and properties
        self._low_cut = low_cut
        self._high_cut = high_cut
        self._fs = fs
        self._order = order
        self._in_type = in_type

        self.properties["shape"] = in_type.shape_str
        self.properties["fs"] = fs
        self.properties["low_cut"] = low_cut
        self.properties["high_cut"] = high_cut
        self.properties["order"] = order

        # 6. Notify graph of updates
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_series(self, values: list[PortValue]):
        input_series: np.ndarray = values[0]
        try:
            filtered, self._zi = lfilter(
                self._b, self._a, input_series, axis=0, zi=self._zi
            )
            # print(f"{input_series.shape=}, {filtered.shape=}, {self._zi.shape=}")
        except Exception as e:
            logger.warning(f"{self.name=}: {e=}")
            traceback.print_exc()
            return

        self._output_series.dispatch([filtered])


class HighPassFilterNode(BaseNode):
    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Butterworth high-pass filter for continuous streaming data",
            category="Filter",
            inputs={"series": ""},
            outputs={"series": ""},
        )

    PROPERTIES = {
        "shape": {"description": "Input shape suffix", "data_type": "str", "default_value": ""},
        "fs": {"description": "Sample rate (Hz)", "data_type": "u32", "default_value": 250},
        "cutoff": {"description": "Cutoff frequency (Hz)", "data_type": "f64", "default_value": 4.0},
        "order": {"description": "Filter order", "data_type": "u32", "default_value": 4},
    }

    def __init__(self, name: str):
        super().__init__(name, "HighPassFilter")

        in_type = PortValueType(DataType.Float32)
        cutoff = 4.0
        fs = 250
        order = 4
        b, a, zi = self._update_filter(fs, cutoff, order, in_type.shape)

        input_series = StaticTypeInputPort("series", [in_type])
        output_series = StaticTypeOutputPort("series", [in_type])

        input_series.subscribe(id(self), self._on_series)

        self._fs = fs
        self._cutoff = cutoff
        self._order = order
        self._b = b
        self._a = a
        self._zi = zi
        self._in_type = in_type

        self._input_series = input_series
        self._output_series = output_series

        self.inputs[input_series.name] = input_series
        self.outputs[output_series.name] = output_series

    def _update_filter(
        self,
        fs: int,
        cutoff: float,
        order: int,
        input_shape: tuple[int, ...],
    ):
        nyquist = 0.5 * fs
        critical_freq = cutoff / nyquist
        b, a = butter(order, critical_freq, btype="high")  # type: ignore
        base_zi = lfilter_zi(b, a)

        if len(input_shape) == 0:
            return b, a, base_zi

        # Fixed to axis=0 (filtering along the time axis)
        axis = 0

        # Convert the nonoha port shape notation to the NumPy form
        np_shape = (1,) + input_shape

        # Compute the final shape of the zi array
        zi_shape = list(np_shape)
        zi_shape[axis] = len(base_zi)

        # Create an empty zi
        zi = np.zeros(zi_shape, dtype=np.float32)

        # Expand base_zi along axis
        reshape_shape = [1] * len(np_shape)
        reshape_shape[axis] = len(base_zi)

        reshaped_zi = base_zi.reshape(reshape_shape)
        expanded_zi = np.broadcast_to(reshaped_zi, zi_shape)
        zi = expanded_zi.copy()

        return b, a, zi

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract and validate properties
        cutoff = self._cutoff
        if "cutoff" in properties:
            cutoff = float(properties["cutoff"])

        fs = self._fs
        if "fs" in properties:
            fs = int(properties["fs"])

        order = self._order
        if "order" in properties:
            order = int(properties["order"])

        in_type = self._in_type
        if "shape" in properties:
            shape_str = properties["shape"]
            in_type = name_to_port_value_type(f"f32{shape_str}")

        # 2. Create new ports if necessary
        new_input_series = self._input_series
        new_output_series = self._output_series
        if in_type != self._in_type:
            new_input_series = StaticTypeInputPort("series", [in_type])
            new_output_series = StaticTypeOutputPort("series", [in_type])

        # 3. Update filter coefficients if necessary
        if (
            cutoff != self._cutoff
            or fs != self._fs
            or order != self._order
            or in_type != self._in_type
        ):
            b, a, zi = self._update_filter(fs, cutoff, order, in_type.shape)
            self._b = b
            self._a = a
            self._zi = zi

        # 4. Replace ports if changed
        if new_input_series != self._input_series:
            replace_node_input_port(
                self, self._input_series, new_input_series, self._on_series
            )
            self._input_series = new_input_series

        if new_output_series != self._output_series:
            replace_node_output_port(self, self._output_series, new_output_series)
            self._output_series = new_output_series

        # 5. Update instance variables and properties
        self._cutoff = cutoff
        self._fs = fs
        self._order = order
        self._in_type = in_type

        self.properties["shape"] = in_type.shape_str
        self.properties["fs"] = fs
        self.properties["cutoff"] = cutoff
        self.properties["order"] = order

        # 6. Notify graph of updates
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_series(self, values: list[PortValue]):
        input_series: np.ndarray = values[0]
        try:
            filtered, self._zi = lfilter(
                self._b, self._a, input_series, axis=0, zi=self._zi
            )
        except Exception as e:
            logger.warning(f"{self.name=}: {e=}")
            traceback.print_exc()
            return

        self._output_series.dispatch([filtered])


class LowPassFilterNode(BaseNode):
    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Butterworth low-pass filter for continuous streaming data",
            category="Filter",
            inputs={"series": ""},
            outputs={"series": ""},
        )

    PROPERTIES = {
        "shape": {"description": "Input shape suffix", "data_type": "str", "default_value": ""},
        "fs": {"description": "Sample rate (Hz)", "data_type": "u32", "default_value": 250},
        "cutoff": {"description": "Cutoff frequency (Hz)", "data_type": "f64", "default_value": 40.0},
        "order": {"description": "Filter order", "data_type": "u32", "default_value": 4},
    }

    def __init__(self, name: str):
        super().__init__(name, "LowPassFilter")

        in_type = PortValueType(DataType.Float32)
        cutoff = 40.0
        fs = 250
        order = 4
        b, a, zi = self._update_filter(fs, cutoff, order, in_type.shape)

        input_series = StaticTypeInputPort("series", [in_type])
        output_series = StaticTypeOutputPort("series", [in_type])

        input_series.subscribe(id(self), self._on_series)

        self._fs = fs
        self._cutoff = cutoff
        self._order = order
        self._b = b
        self._a = a
        self._zi = zi
        self._in_type = in_type

        self._input_series = input_series
        self._output_series = output_series

        self.inputs[input_series.name] = input_series
        self.outputs[output_series.name] = output_series

    def _update_filter(
        self,
        fs: int,
        cutoff: float,
        order: int,
        input_shape: tuple[int, ...],
    ):
        nyquist = 0.5 * fs
        critical_freq = cutoff / nyquist
        b, a = butter(order, critical_freq, btype="low")  # type: ignore
        base_zi = lfilter_zi(b, a)

        if len(input_shape) == 0:
            return b, a, base_zi

        # Fixed to axis=0 (filtering along the time axis)
        axis = 0

        # Convert the nonoha port shape notation to the NumPy form
        np_shape = (1,) + input_shape

        # Compute the final shape of the zi array
        zi_shape = list(np_shape)
        zi_shape[axis] = len(base_zi)

        # Create an empty zi
        zi = np.zeros(zi_shape, dtype=np.float32)

        # Expand base_zi along axis
        reshape_shape = [1] * len(np_shape)
        reshape_shape[axis] = len(base_zi)

        reshaped_zi = base_zi.reshape(reshape_shape)
        expanded_zi = np.broadcast_to(reshaped_zi, zi_shape)
        zi = expanded_zi.copy()

        return b, a, zi

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract and validate properties
        cutoff = self._cutoff
        if "cutoff" in properties:
            cutoff = float(properties["cutoff"])

        fs = self._fs
        if "fs" in properties:
            fs = int(properties["fs"])

        order = self._order
        if "order" in properties:
            order = int(properties["order"])

        in_type = self._in_type
        if "shape" in properties:
            shape_str = properties["shape"]
            in_type = name_to_port_value_type(f"f32{shape_str}")

        # 2. Create new ports if necessary
        new_input_series = self._input_series
        new_output_series = self._output_series
        if in_type != self._in_type:
            new_input_series = StaticTypeInputPort("series", [in_type])
            new_output_series = StaticTypeOutputPort("series", [in_type])

        # 3. Update filter coefficients if necessary
        if (
            cutoff != self._cutoff
            or fs != self._fs
            or order != self._order
            or in_type != self._in_type
        ):
            b, a, zi = self._update_filter(fs, cutoff, order, in_type.shape)
            self._b = b
            self._a = a
            self._zi = zi

        # 4. Replace ports if changed
        if new_input_series != self._input_series:
            replace_node_input_port(
                self, self._input_series, new_input_series, self._on_series
            )
            self._input_series = new_input_series

        if new_output_series != self._output_series:
            replace_node_output_port(self, self._output_series, new_output_series)
            self._output_series = new_output_series

        # 5. Update instance variables and properties
        self._cutoff = cutoff
        self._fs = fs
        self._order = order
        self._in_type = in_type

        self.properties["shape"] = in_type.shape_str
        self.properties["fs"] = fs
        self.properties["cutoff"] = cutoff
        self.properties["order"] = order

        # 6. Notify graph of updates
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_series(self, values: list[PortValue]):
        input_series: np.ndarray = values[0]
        try:
            filtered, self._zi = lfilter(
                self._b, self._a, input_series, axis=0, zi=self._zi
            )
        except Exception as e:
            logger.warning(f"{self.name=}: {e=}")
            traceback.print_exc()
            return

        self._output_series.dispatch([filtered])


class NotchFilterNode(BaseNode):
    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="IIR notch filter to remove a specific frequency",
            category="Filter",
            inputs={"series": ""},
            outputs={"series": ""},
        )

    PROPERTIES = {
        "shape": {"description": "Input shape suffix", "data_type": "str", "default_value": ""},
        "fs": {"description": "Sample rate (Hz)", "data_type": "u32", "default_value": 600},
        "f_notch": {"description": "Notch frequency (Hz)", "data_type": "f64", "default_value": 60.0},
        "bw": {"description": "Bandwidth (Hz)", "data_type": "f64", "default_value": 2.5},
    }

    def __init__(self, name: str):
        super().__init__(name, "NotchFilter")

        in_type = PortValueType(DataType.Float32)
        fs = 600
        f_notch = 60.0
        bw = 2.5
        b, a, zi = self._update_filter(f_notch, bw, fs=fs, input_shape=in_type.shape)

        input_series = StaticTypeInputPort("series", [in_type])
        output_series = StaticTypeOutputPort("series", [in_type])

        input_series.subscribe(id(self), self._on_series)

        self._fs = fs
        self._f_notch = f_notch
        self._bw = bw
        self._b = b
        self._a = a
        self._zi = zi
        self._in_type = in_type

        self._input_series = input_series
        self._output_series = output_series

        self.inputs[input_series.name] = input_series
        self.outputs[output_series.name] = output_series

    def _update_filter(
        self,
        f0: float,
        bw: float,
        fs: float,
        input_shape: tuple[int, ...],
    ):
        Q = f0 / bw
        b, a = iirnotch(f0, Q, fs)
        base_zi = lfilter_zi(b, a)

        if len(input_shape) == 0:
            return b, a, base_zi

        # Fixed to axis=0 (filtering along the time axis)
        axis = 0

        # Convert the nonoha port shape notation to the NumPy form.
        # e.g. (3,) in port notation is actually an (n, 3) NumPy array;
        # the first (sample count) dimension is provisionally set to 1.
        np_shape = (1,) + input_shape

        # Compute the final shape of the zi array
        zi_shape = list(np_shape)
        zi_shape[axis] = len(base_zi)  # along axis: filter length - 1

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

        return b, a, zi

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract and validate properties
        f_notch = self._f_notch
        if "f_notch" in properties:
            f_notch = float(properties["f_notch"])

        bw = self._bw
        if "bw" in properties:
            bw = float(properties["bw"])

        fs = self._fs
        if "fs" in properties:
            fs = int(properties["fs"])

        Q = f_notch / bw

        in_type = self._in_type
        if "shape" in properties:
            shape_str = properties["shape"]
            in_type = name_to_port_value_type(f"f32{shape_str}")

        # 2. Create new ports if necessary
        new_input_series = self._input_series
        new_output_series = self._output_series
        if in_type != self._in_type:
            new_input_series = StaticTypeInputPort("series", [in_type])
            new_output_series = StaticTypeOutputPort("series", [in_type])

        # 3. Update filter coefficients if necessary
        if (
            f_notch != self._f_notch
            or bw != self._bw
            or fs != self._fs
            or in_type != self._in_type
        ):
            b, a, zi = self._update_filter(f_notch, bw, fs=fs, input_shape=in_type.shape)
            self._b = b
            self._a = a
            self._zi = zi

        # 4. Replace ports if changed
        if new_input_series != self._input_series:
            replace_node_input_port(
                self, self._input_series, new_input_series, self._on_series
            )
            self._input_series = new_input_series

        if new_output_series != self._output_series:
            replace_node_output_port(self, self._output_series, new_output_series)
            self._output_series = new_output_series

        # 5. Update instance variables and properties
        self._f_notch = f_notch
        self._bw = bw
        self._fs = fs
        self._in_type = in_type

        self.properties["shape"] = in_type.shape_str
        self.properties["fs"] = fs
        self.properties["f_notch"] = f_notch
        self.properties["bw"] = bw

        # 6. Notify graph of updates
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_series(self, values: list[PortValue]):
        input_series: np.ndarray = values[0]
        try:
            filtered, self._zi = lfilter(
                self._b, self._a, input_series, axis=0, zi=self._zi
            )
        except Exception as e:
            logger.warning(f"{self.name=}: {e=}")
            traceback.print_exc()
            return

        self._output_series.dispatch([filtered])


class SmaFilterNode(BaseNode):
    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Simple moving average filter for smoothing streaming data",
            category="Filter",
            inputs={"series": "", "reset": ""},
            outputs={"series": ""},
        )

    PROPERTIES = {
        "in_type": {"description": "Input port value type", "data_type": "str", "default_value": "f32"},
        "win_size": {"description": "Window size", "data_type": "u32", "default_value": 10},
    }

    def __init__(self, name: str):
        super().__init__(name, "SmaFilter")

        in_type = PortValueType(DataType.Float32)
        win_size = 10
        b, a, zi = self._update_filter(win_size, in_type.shape)

        input_series = StaticTypeInputPort("series", [in_type])
        output_series = StaticTypeOutputPort("series", [in_type])
        input_reset = StaticTypeInputPort("reset", [PortValueType(DataType.Any)])

        input_series.subscribe(id(self), self._on_input_series)
        input_reset.subscribe(id(self), self._on_input_reset)

        self._win_size = win_size
        self._b = b
        self._a = a
        self._zi = zi
        self._in_type = in_type

        self._input_series = input_series
        self._input_reset = input_reset
        self._output_series = output_series

        self.inputs[input_series.name] = input_series
        self.inputs[input_reset.name] = input_reset
        self.outputs[output_series.name] = output_series

    def _update_filter(
        self,
        win_size: int,
        input_shape: tuple[int, ...],
    ):
        # Compute the FIR filter coefficients
        b = np.ones(win_size) / win_size  # impulse response of the moving average
        a = [1.0]  # FIR filter, so a is just 1
        base_zi = np.zeros(len(b) - 1)

        if len(input_shape) == 0:
            return b, a, base_zi

        # Fixed to axis=0 (filtering along the time axis)
        axis = 0

        # Build the zi array the same way as BandPassFilterNode
        np_shape = (1,) + input_shape
        zi_shape = list(np_shape)
        zi_shape[axis] = len(base_zi)

        zi = np.zeros(zi_shape, dtype=np.float32)

        # Expand base_zi along axis
        reshape_shape = [1] * len(np_shape)
        reshape_shape[axis] = len(base_zi)

        reshaped_zi = base_zi.reshape(reshape_shape)
        expanded_zi = np.broadcast_to(reshaped_zi, zi_shape)
        zi = expanded_zi.copy()

        return b, a, zi

    def set_properties(self, properties: dict[str, PropertyDataType]):
        win_size = self._win_size
        if "win_size" in properties:
            win_size = int(properties["win_size"])
            if win_size < 1:
                raise ValueError("Window size must be at least 1")

        in_type = self._in_type
        if "in_type" in properties:
            in_type_str = properties["in_type"]
            if not isinstance(in_type_str, str):
                raise ValueError("in_type must be a string")
            in_type = name_to_port_value_type(in_type_str)

            if in_type.data_type not in [DataType.Float32, DataType.Float64]:
                raise ValueError("In type must be Float32 or Float64")

        new_input_series = self._input_series
        if in_type != self._in_type:
            new_input_series = StaticTypeInputPort("series", [in_type])

        new_output_series = self._output_series
        if in_type != self._in_type:
            new_output_series = StaticTypeOutputPort("series", [in_type])

        if win_size != self._win_size or in_type != self._in_type:
            # Update the filter coefficients
            b, a, zi = self._update_filter(win_size, in_type.shape)
            self._b = b
            self._a = a
            self._zi = zi

        # Replace ports
        if new_input_series != self._input_series:
            replace_node_input_port(
                self, self._input_series, new_input_series, self._on_input_series
            )
            self._input_series = new_input_series

        if new_output_series != self._output_series:
            replace_node_output_port(self, self._output_series, new_output_series)
            self._output_series = new_output_series

        self._win_size = win_size
        self._in_type = in_type

        self.properties["in_type"] = in_type.name
        self.properties["win_size"] = win_size

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_input_reset(self, values: list[PortValue]):
        """Reset SMA filter state to zero"""
        self._zi.fill(0.0)

    def _on_input_series(self, values: list[PortValue]):
        series: np.ndarray = values[0]

        try:
            filtered, self._zi = lfilter(self._b, self._a, series, axis=0, zi=self._zi)
        except Exception as e:
            logger.warning(f"{self.name=}: {e=}")
            traceback.print_exc()
            return

        self._output_series.dispatch([filtered])


class ResamplePolyNode(BaseNode):
    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Polyphase resampling for sample rate conversion of streaming data",
            category="Filter",
            inputs={"series": "", "reset": ""},
            outputs={"series": ""},
        )

    PROPERTIES = {
        "in_type": {"description": "Input port value type", "data_type": "str", "default_value": "f32"},
        "fs_in": {"description": "Input sample rate (Hz)", "data_type": "u32", "default_value": 250},
        "fs_out": {"description": "Output sample rate (Hz)", "data_type": "u32", "default_value": 600},
        "overlap_in": {"description": "Overlap buffer size (samples)", "data_type": "u32", "default_value": 128},
    }

    """Stateful polyphase resampling node for continuous streaming.

    Uses scipy.signal.resample_poly with padding buffer to ensure smooth
    transitions at chunk boundaries. Pre-calculates up/down ratios to avoid
    redundant GCD calculations.

    Implementation note: scipy's resample_poly recalculates the filter on each
    call, which is a known limitation. However, this ensures correctness with
    minimal performance impact for typical use cases.
    """

    def __init__(self, name: str):
        super().__init__(name, "ResamplePoly")

        in_type = PortValueType(DataType.Float32)
        fs_in = 250
        fs_out = 600

        input_series = StaticTypeInputPort("series", [in_type])
        output_series = StaticTypeOutputPort("series", [in_type])
        input_reset = StaticTypeInputPort("reset", [PortValueType(DataType.Any)])

        input_series.subscribe(id(self), self._on_input_series)
        input_reset.subscribe(id(self), self._on_input_reset)

        # Pre-calculate resampling parameters and filter
        up, down, h = self._calc_resample_params(fs_in, fs_out)

        # State variables
        self._fs_in = fs_in
        self._fs_out = fs_out
        self._up = up
        self._down = down
        self._h = h  # Pre-calculated anti-aliasing filter
        self._in_type = in_type
        self._overlap_in = 128  # Default overlap for symmetric processing
        self._overlap_out = self._overlap_in * self._fs_out // self._fs_in

        # Initialize buffer with correct shape for multi-dimensional data
        dtype = np.float32 if in_type.data_type == DataType.Float32 else np.float64
        if len(in_type.shape) == 0:
            # Scalar: data shape is (N,), buffer shape is (0,)
            self._pad_buffer = np.array([], dtype=dtype)
        else:
            # Multi-dimensional: data shape is (N, C), buffer shape is (0, C)
            self._pad_buffer = np.zeros((0,) + in_type.shape, dtype=dtype)

        self._input_series = input_series
        self._output_series = output_series
        self._input_reset = input_reset

        self.inputs[input_series.name] = input_series
        self.inputs[input_reset.name] = input_reset
        self.outputs[output_series.name] = output_series

    def _calc_resample_params(
        self, fs_in: int, fs_out: int
    ) -> tuple[int, int, np.ndarray]:
        """Calculate resampling parameters and anti-aliasing filter.

        Returns:
            up: Upsampling factor
            down: Downsampling factor
            h: FIR filter coefficients (gain-compensated for upsampling)
        """
        gcd = np.gcd(fs_in, fs_out)
        up = fs_out // gcd
        down = fs_in // gcd

        # Design anti-aliasing filter (same as scipy.signal.resample_poly)
        max_rate = max(up, down)
        f_c = 1.0 / max_rate  # Normalized cutoff
        half_len = 10 * max_rate
        h = firwin(2 * half_len + 1, f_c, window=("kaiser", 5.0))

        # Scale by upsampling factor for correct gain
        h = h * up

        return up, down, h

    def set_properties(self, properties: dict[str, PropertyDataType]):
        """Update resampling parameters and recalculate filter if needed."""
        # 1. Extract properties
        fs_in = self._fs_in
        fs_out = self._fs_out
        in_type = self._in_type
        overlap_in = self._overlap_in

        if "fs_in" in properties:
            fs_in = int(properties["fs_in"])
            if fs_in <= 0:
                raise ValueError("fs_in must be positive")

        if "fs_out" in properties:
            fs_out = int(properties["fs_out"])
            if fs_out <= 0:
                raise ValueError("fs_out must be positive")

        if "overlap_in" in properties:
            overlap_in = int(properties["overlap_in"])
            if overlap_in <= 0:
                raise ValueError("overlap_in must be positive")

        if "in_type" in properties:
            in_type_str = properties["in_type"]
            if not isinstance(in_type_str, str):
                raise ValueError("in_type must be a string")
            in_type = name_to_port_value_type(in_type_str)

            if in_type.data_type not in [DataType.Float32, DataType.Float64]:
                raise ValueError("In type must be Float32 or Float64")

        # 2. Check if ports need to be recreated
        new_input_series = self._input_series
        new_output_series = self._output_series

        if in_type != self._in_type:
            new_input_series = StaticTypeInputPort("series", [in_type])
            new_output_series = StaticTypeOutputPort("series", [in_type])

        # 3. Recalculate filter if sampling rates changed
        if fs_in != self._fs_in or fs_out != self._fs_out:
            up, down, h = self._calc_resample_params(fs_in, fs_out)
            self._up = up
            self._down = down
            self._h = h

            # Reset buffers when parameters change
            dtype = np.float32 if in_type.data_type == DataType.Float32 else np.float64
            if len(in_type.shape) == 0:
                self._pad_buffer = np.array([], dtype=dtype)
            else:
                self._pad_buffer = np.zeros((0,) + in_type.shape, dtype=dtype)

        if (
            fs_in != self._fs_in
            or fs_out != self._fs_out
            or overlap_in != self._overlap_in
        ):
            self._overlap_out = overlap_in * fs_out // fs_in

        # 4. Replace ports if needed
        if new_input_series != self._input_series:
            replace_node_input_port(
                self, self._input_series, new_input_series, self._on_input_series
            )
            self._input_series = new_input_series

        if new_output_series != self._output_series:
            replace_node_output_port(self, self._output_series, new_output_series)
            self._output_series = new_output_series

        # 5. Update state
        self._fs_in = fs_in
        self._fs_out = fs_out
        self._in_type = in_type
        self._overlap_in = overlap_in

        self.properties["in_type"] = in_type.name
        self.properties["fs_in"] = fs_in
        self.properties["fs_out"] = fs_out
        self.properties["overlap_in"] = overlap_in

        # 6. Notify graph
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_input_reset(self, values: list[PortValue]):
        """Reset resampling state buffers."""
        dtype = (
            np.float32 if self._in_type.data_type == DataType.Float32 else np.float64
        )
        # Reset to empty buffer with correct shape
        if len(self._in_type.shape) == 0:
            self._pad_buffer = np.array([], dtype=dtype)
        else:
            self._pad_buffer = np.zeros((0,) + self._in_type.shape, dtype=dtype)

    def _on_input_series(self, values: list[PortValue]):

        # start_time = time.perf_counter()

        input_series: np.ndarray = values[0]

        samples = np.concatenate([self._pad_buffer, input_series], axis=0)
        if len(samples) < self._overlap_in * 2:
            self._pad_buffer = samples
            # elapsed = (time.perf_counter() - start_time) * 1000
            # print(
            #     f"ResamplePoly: {elapsed:.3f}ms (buffering, {len(samples)} samples)"
            # )
            return

        # === Processing identical to scipy.resample_poly ===
        # Zero-pad filter to put output samples at center (same as scipy)
        n_in = len(samples)
        n_out = n_in * self._up
        n_out = n_out // self._down + bool(n_out % self._down)

        half_len = (len(self._h) - 1) // 2  # Original filter half-length
        n_pre_pad = self._down - half_len % self._down
        n_post_pad = 0
        n_pre_remove = (half_len + n_pre_pad) // self._down

        # Ensure output is long enough
        # Calculate expected output length using scipy's upfirdn formula
        # Reference: https://github.com/scipy/scipy/blob/main/scipy/signal/_upfirdn_apply.pyx
        # Formula: ((input_len - 1) * up + filter_len + down - 1) // down
        filter_len = len(self._h) + n_pre_pad + n_post_pad
        while (
            (n_in - 1) * self._up + filter_len + self._down - 1
        ) // self._down < n_out + n_pre_remove:
            n_post_pad += 1
            filter_len = len(self._h) + n_pre_pad + n_post_pad

        # print(f"{n_pre_pad=}, {n_post_pad=}")

        # Pad filter
        h_padded = np.concatenate(
            [
                np.zeros(n_pre_pad, dtype=self._h.dtype),
                self._h,
                np.zeros(n_post_pad, dtype=self._h.dtype),
            ]
        )

        # Apply upfirdn
        y = upfirdn(h_padded, samples, self._up, self._down, axis=0)

        # Trim output (center extraction)
        n_pre_remove_end = n_pre_remove + n_out
        resamples = y[n_pre_remove:n_pre_remove_end]
        # === End of the scipy.resample_poly-identical processing ===

        self._pad_buffer = samples[-2 * self._overlap_in :]

        # elapsed = (time.perf_counter() - start_time) * 1000
        # print(
        #     f"ResamplePoly: {elapsed:.3f}ms (in={len(input_series)}, total={len(samples)}, out={len(resamples[self._overlap_out : -self._overlap_out])})"
        # )

        # except Exception as e:
        #     print(f"{e=}")
        #     traceback.print_exc()

        self._output_series.dispatch(
            [resamples[self._overlap_out : -self._overlap_out]]
        )


# class MedianBasedOutlierFilterNode(BaseNode):
#     def __init__(self, name: str):
#         super().__init__(name, "MedianBasedOutlierFilter")

#         win_size = 100
#         threshold = 3.0  # MADの何倍を外れ値とするか

#         input_series = StaticTypePort("series", PortDirection.INPUT, [DataType.Float32])
#         input_series.subscribe(id(self), self._on_input_series)

#         input_reset = StaticTypePort("reset", PortDirection.INPUT, [DataType.Any])
#         input_reset.subscribe(id(self), self._on_input_reset)

#         output_series = StaticTypePort(
#             "series", PortDirection.OUTPUT, [DataType.Float32]
#         )

#         self._win_size = win_size
#         self._threshold = threshold
#         self._index = 0
#         self._buffer_full = False
#         self._buffer = np.zeros(win_size, dtype=np.float32)
#         self._input_series = input_series
#         self._output_series = output_series

#         self.inputs[input_series.name] = input_series
#         self.inputs[input_reset.name] = input_reset
#         self.outputs[output_series.name] = output_series

#         self.properties["win_size"] = win_size
#         self.properties["threshold"] = threshold

#     def set_properties(self, properties: dict[str, PropertyDataType]):
#         win_size = self._win_size
#         threshold = self._threshold

#         if "win_size" in properties:
#             win_size = int(properties["win_size"])
#         if "threshold" in properties:
#             threshold = float(properties["threshold"])

#         if win_size != self._win_size:
#             self._buffer = np.zeros(win_size, dtype=np.float32)
#             self._index = 0
#             self._buffer_full = False

#         self._win_size = win_size
#         self._threshold = threshold

#         self.properties["win_size"] = win_size
#         self.properties["threshold"] = threshold

#         if self.parent_graph is not None:
#             self.parent_graph.notify_properties_changed(self)

#     def _on_input_reset(self, values: list[Any]):
#         self._index = 0
#         self._buffer_full = False

#     def _on_input_series(self, values: list[Any]):
#         series: np.ndarray = values[0]

#         filtered = np.zeros_like(series)

#         for index, x in enumerate(series):
#             if self._buffer_full:
#                 buffer_data = self._buffer
#             elif self._index > 0:
#                 buffer_data = self._buffer[: self._index]
#             else:
#                 filtered[index] = x
#                 self._buffer[self._index] = x
#                 self._index += 1
#                 if self._index >= self._win_size:
#                     self._buffer_full = True
#                     self._index = 0
#                 continue

#             median = np.median(buffer_data)
#             mad = np.median(np.abs(buffer_data - median))

#             # 外れ値判定と置換
#             filtered[index] = (
#                 median if np.abs(x - median) > self._threshold * mad else x
#             )

#             self._buffer[self._index] = x
#             self._index += 1
#             if self._index >= self._win_size:
#                 self._buffer_full = True
#                 self._index = 0

#         # print(f"{filtered=}")
#         self._output_series.dispatch([filtered])


# class CompressorNode(BaseNode):
#     def __init__(self, name: str):
#         super().__init__(name, "Compressor")

#         input_series = StaticTypePort("series", PortDirection.INPUT, [DataType.Float32])
#         input_series.subscribe(id(self), self._on_input_series)

#         output_series = StaticTypePort(
#             "series", PortDirection.OUTPUT, [DataType.Float32]
#         )

#         threshold = 0.5
#         ratio = 0.5
#         gain = 1.0 / (threshold + ratio * (1 - threshold))

#         self._threshold = threshold
#         self._ratio = ratio
#         self._gain = gain

#         self._input_series = input_series
#         self._output_series = output_series

#         self.inputs[input_series.name] = input_series
#         self.outputs[output_series.name] = output_series

#         self.properties["threshold"] = threshold
#         self.properties["ratio"] = ratio

#     def set_properties(self, properties: dict[str, PropertyDataType]):
#         threshold = self._threshold
#         ratio = self._ratio

#         if "threshold" in properties:
#             threshold = float(properties["threshold"])
#         if "ratio" in properties:
#             ratio = float(properties["ratio"])

#         gain = 1.0 / (threshold + ratio * (1 - threshold))

#         self._threshold = threshold
#         self._ratio = ratio
#         self._gain = gain

#         self.properties["threshold"] = threshold
#         self.properties["ratio"] = ratio

#         if self.parent_graph is not None:
#             self.parent_graph.notify_properties_changed(self)

#     def _on_input_series(self, values: list[Any]):
#         series: np.ndarray = values[0]

#         # filtered = np.zeros_like(series)

#         # for index, x in enumerate(series):
#         #     filtered[index] = (
#         #         (self._threshold + (x - self._threshold) * self._ratio)
#         #         if x > self._threshold
#         #         else x
#         #     )
#         # 正の値と負の値の両方に対して圧縮処理を適用
#         filtered = np.where(
#             series > self._threshold,  # 正の値でしきい値を超える場合
#             self._threshold
#             + (series - self._threshold) * self._ratio,  # 正の値の圧縮処理
#             np.where(
#                 series < -self._threshold,  # 負の値でしきい値を下回る場合
#                 -self._threshold
#                 + (series + self._threshold) * self._ratio,  # 負の値の圧縮処理
#                 series,  # しきい値の範囲内(-threshold～threshold)はそのまま
#             ),
#         )

#         filtered *= self._gain

#         self._output_series.dispatch([filtered])
