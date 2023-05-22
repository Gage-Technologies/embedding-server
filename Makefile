install-server:
	cd server && make install

install-integration-tests:
	cd integration-tests && pip install -r requirements.txt

install-router:
	cd router && cargo install --path .

install-launcher:
	cd launcher && cargo install --path .

install-benchmark:
	cd benchmark && cargo install --path .

install: install-server install-router install-launcher

server-dev:
	cd server && make run-dev

router-dev:
	cd router && cargo run -- --port 8080

rust-tests: install-router install-launcher
	cargo test

integration-tests: install-integration-tests
	pytest -s -vv -m "not private" integration-tests

update-integration-tests: install-integration-tests
	pytest -s -vv --snapshot-update integration-tests

python-server-tests:
	HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv -m "not private" server/tests

python-client-tests:
	pytest clients/python/tests

python-tests: python-server-tests python-client-tests

run-bloom-560m:
	embedding-server-launcher --model-id all-MiniLM-L6-v2 --num-shard 2 --port 8080

run-bloom-560m-quantize:
	embedding-server-launcher --model-id all-MiniLM-L6-v2 --num-shard 2 --quantize --port 8080

download-bloom:
	HF_HUB_ENABLE_HF_TRANSFER=1 embedding-server download-weights bigscience/bloom

run-bloom:
	embedding-server-launcher --model-id bigscience/bloom --num-shard 8 --port 8080

run-bloom-quantize:
	embedding-server-launcher --model-id bigscience/bloom --num-shard 8 --quantize --port 8080

push-oss:
	git checkout oss
	git push upstream oss:main