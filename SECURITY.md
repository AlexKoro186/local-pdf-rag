# Security policy

## Intended deployment

This project is designed for a single user on a trusted local machine. The Streamlit server
does not provide authentication, tenant isolation, or document-level authorization. Do not
expose it directly to an untrusted network.

## Sensitive data

PDFs, the Chroma index, `.env`, Streamlit secrets, and local model data are excluded from Git.
Review staged files before every public push:

```bash
git status --short
git diff --cached --name-only
```

Deleting a source PDF does not automatically delete its previously indexed vectors. Clear
or rebuild the local `chroma_db/` directory when removing sensitive material.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature. Include the affected version,
reproduction steps, impact, and any suggested mitigation. Avoid attaching private source
documents or secrets.
