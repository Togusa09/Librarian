#!/bin/sh
set -e

# Ensure configured mount points exist so user-mounted volumes can be used
mkdir -p "${DOCUMENT_ARCHIVE_PATH}" "${DATA_PATH}"

# Best-effort permissions change (may be a no-op on some platforms)
chown -R nobody:nogroup "${DOCUMENT_ARCHIVE_PATH}" "${DATA_PATH}" 2>/dev/null || true

exec "$@"
