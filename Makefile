PYTHON ?= python3

.PHONY: bootstrap build build-verifier build-table-builder test verifier-test pair56-smoke pair34-smoke validate doctor verify-package audit-tree prepare-pair34 prepare-pair56 verify-pair34 verify-pair56 verify-all figures clean

bootstrap:
	./environment/fetch_upstream.sh

build: bootstrap build-verifier build-table-builder
	./environment/build.sh

build-verifier: bootstrap
	./environment/build_verifier.sh

build-table-builder: bootstrap
	./environment/build_table_builder.sh

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

verifier-test: build-verifier
	MDR_REQUIRE_GO_VERIFIER=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_verifier*.py' -v

pair56-smoke:
	./scripts/train_pair56 --smoke --device cpu --amp fp32 --output-dir /tmp/mdr-pair56-smoke --run-id make-smoke
	./scripts/test_pair56 --checkpoint /tmp/mdr-pair56-smoke/pair56-qmlp_make-smoke_best.pt --device cpu --amp fp32 --tests 0 --val-size 128 --val-batch-size 64

pair34-smoke:
	./scripts/train_pair34 --smoke --device cpu --amp fp32 --output-dir /tmp/mdr-pair34-smoke --run-id make-smoke

validate:
	PYTHONPATH=src $(PYTHON) -m mdr.cli validate-config

doctor:
	PYTHONPATH=src $(PYTHON) -m mdr.cli doctor

verify-package:
	$(PYTHON) scripts/verify_publication_artifacts.py

audit-tree:
	$(PYTHON) scripts/audit_publication_tree.py

prepare-pair34:
	$(PYTHON) scripts/unpack_publication_certificate.py pair34

prepare-pair56:
	$(PYTHON) scripts/unpack_publication_certificate.py pair56

verify-pair34:
	./scripts/reproduce_pair pair34

verify-pair56:
	./scripts/reproduce_pair pair56

verify-all: verify-pair34 verify-pair56

figures:
	$(PYTHON) scripts/generate_publication_figures.py

clean:
	rm -rf build/upstream-bin build/*.tmp artifacts/reports
