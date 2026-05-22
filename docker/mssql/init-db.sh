#!/usr/bin/env bash
set -euo pipefail

SQLCMD="/opt/mssql-tools18/bin/sqlcmd"
if [ ! -x "$SQLCMD" ]; then
  SQLCMD="/opt/mssql-tools/bin/sqlcmd"
fi

DATABASE_NAME="${MSSQL_DATABASE:-cartao_resposta}"

"$SQLCMD" \
  -C \
  -S mssql \
  -U sa \
  -P "${MSSQL_SA_PASSWORD}" \
  -Q "IF DB_ID(N'${DATABASE_NAME}') IS NULL EXEC(N'CREATE DATABASE [${DATABASE_NAME}]');"
