# HTTP API

```bash
pip install -e .[api]
qudock serve --port 8000          # interactive docs at http://127.0.0.1:8000/docs
```

The screener is trained on BACE and BBBP on the first request and kept in memory.

```bash
curl -X POST http://127.0.0.1:8000/screen \
     -H "Content-Type: application/json" \
     -d '{"smiles": ["CC(=O)Nc1ccc(O)cc1", "NC(=N)c1ccccc1"], "top": 10}'
```

The response lists `molecules` ranked best first, with the same columns as
[`qudock screen`](screening.md), and `invalid` for SMILES RDKit could not parse.
Up to 1,000 molecules per request. `GET /health` reports whether the model is
trained yet.
