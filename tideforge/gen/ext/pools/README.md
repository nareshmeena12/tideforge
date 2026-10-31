Committed manifests of the offline trajectory pools (tideforge/gen/ext/_pool.py). The pools themselves live in
$TF_DATA/pools/<name>/ and are never committed. The builders (tools/build_ode_pool.py, tools/build_loop_pool.py) write the
manifest next to the pool ($TF_DATA/pools/<name>/manifest.json) and print its path, so a build never dirties the
checkout; it is committed here as <name>.json. Generation refuses any pool whose files
differ from its committed manifest, and any pool without one.
