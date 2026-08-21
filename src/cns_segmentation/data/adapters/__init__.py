"""Per-dataset adapters that materialize foreign-format datasets into the
spine-generic BIDS-derivatives shape `create_datalist()` expects.

See `base.py` for the shared `prepare(force: bool = False) -> DatasetSpec`
contract each adapter module implements.
"""
