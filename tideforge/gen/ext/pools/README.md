Committed manifests of the offline trajectory pools (tideforge/gen/ext/_pool.py). The pools themselves live in
$TF_DATA/pools/<name>/ and are never committed. The builders (tools/build_ode_pool.py, tools/build_loop_pool.py) write the
manifest here, in the checkout that ran them; commit it from the Perlmutter build. Generation refuses any pool whose files
differ from its committed manifest, and any pool without one.
