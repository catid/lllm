# Setup

```bash
openssl genpkey -algorithm Ed25519 -out ed25519_key.pem

openssl req -new -x509 -key ed25519_key.pem -out ed25519_cert.pem -days 365 \
    -subj "/C=US/ST=YourState/L=YourCity/O=YourOrganization/OU=YourUnit/CN=orchestrator.example.com" \
    -addext "subjectAltName=DNS:orchestrator.example.com,IP:127.0.0.1,IP:192.168.1.100" \
    -addext "keyUsage=critical,digitalSignature" \
    -addext "extendedKeyUsage=serverAuth"
```
