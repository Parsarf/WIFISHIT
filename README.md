# Wi-Fi CSI Sensing Stack

## Quick Start

```bash
./scripts/bootstrap.sh
```

Run the backend server:

```bash
./scripts/run_server.sh
```

Run tests:

```bash
./scripts/run_tests.sh
```

## Developer Commands

```bash
make install
make test
make run-server
```

If your system Python blocks global package installs, use the virtual environment created by `./scripts/bootstrap.sh`.

## CI

GitHub Actions runs `pytest` on Python 3.11 and 3.12 using `.github/workflows/ci.yml`.
