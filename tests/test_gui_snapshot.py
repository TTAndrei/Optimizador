import os

import numpy as np
import pytest

from gui.snapshot import SnapshotReader, SnapshotWriter


def test_snapshot_roundtrip_y_versionado():
    X, Y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(-1, 1, 3))
    writer = SnapshotWriter(f"test_curvo_{os.getpid()}_{id(X)}")
    reader = None
    try:
        writer.create(X, Y)
        reader = SnapshotReader(writer.names)
        inicial = reader.read()
        assert inicial is not None
        assert inicial.sequence == 1
        u, v, p = X[:-1, :-1] * 3, Y[:-1, :-1] * 4, X[:-1, :-1] - Y[:-1, :-1]
        writer.publish(X + 1, Y + 2, u, v, p,
                   {"iter": 7, "t": 0.14, "Cl": 0.32,
                "Cd": 0.021, "it_s": 8.5})
        snapshot = reader.read()
        assert snapshot is not None
        assert snapshot.sequence == 2
        assert snapshot.metrics["iter"] == 7
        assert snapshot.metrics["Cl"] == pytest.approx(0.32)
        assert snapshot.metrics["Cd"] == pytest.approx(0.021)
        assert snapshot.metrics["it_s"] == pytest.approx(8.5)
        np.testing.assert_array_equal(snapshot.p, p)
        assert reader.read() is None
    finally:
        if reader is not None:
            reader.close()
        writer.close()
