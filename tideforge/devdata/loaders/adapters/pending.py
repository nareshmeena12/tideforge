"""Placeholder adapter for sources whose file layout is not confirmed yet (PVDAQ, EEA). `ingest probe` shows the listing; the
real adapter is written from that output. Until then ingesting such a source stops here."""


def read_files(src, files):
    raise NotImplementedError(f"{src.id}: the adapter is not written yet: run `ingest probe --ids {src.id}` and paste the output")
