#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUT_DIR="./output"
QMD_FILE="./interview_deck.qmd"

mkdir -p "$OUT_DIR"

is_wsl() {
  [ -n "${WSL_DISTRO_NAME:-}" ] || grep -qi microsoft /proc/version 2>/dev/null
}

resolve_windows_cmd_in_wsl() {
  local cmd_name="$1"
  if ! is_wsl; then
    return 1
  fi
  if ! command -v powershell.exe >/dev/null 2>&1; then
    return 1
  fi

  local win_path
  win_path="$(powershell.exe -NoProfile -Command \
    "(Get-Command ${cmd_name} -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)" \
    2>/dev/null | tr -d '\r')"

  if [ -z "$win_path" ]; then
    return 1
  fi
  if ! command -v wslpath >/dev/null 2>&1; then
    return 1
  fi

  wslpath -u "$win_path"
}

resolve_cmd() {
  local cmd_name="$1"
  local resolved=""
  if resolved="$(command -v "$cmd_name" 2>/dev/null)"; then
    printf '%s\n' "$resolved"
    return 0
  fi
  if resolved="$(resolve_windows_cmd_in_wsl "$cmd_name" 2>/dev/null)"; then
    printf '%s\n' "$resolved"
    return 0
  fi
  return 1
}

print_cmd_status() {
  local label="$1"
  local cmd_path="${2:-}"
  if [ -n "$cmd_path" ]; then
    local version_line
    version_line="$("$cmd_path" --version 2>/dev/null | head -n 1 || true)"
    if [ -n "$version_line" ]; then
      echo "[check] $label : $version_line"
    else
      echo "[check] $label : $cmd_path"
    fi
  else
    echo "[check] $label : NOT FOUND"
  fi
}

echo "[check] project dir: $ROOT_DIR"
echo "[check] output dir : $OUT_DIR"

QUARTO_CMD="$(resolve_cmd quarto || true)"
PANDOC_CMD="$(resolve_cmd pandoc || true)"
XELATEX_CMD="$(resolve_cmd xelatex || true)"

print_cmd_status "quarto " "$QUARTO_CMD"
print_cmd_status "pandoc " "$PANDOC_CMD"
print_cmd_status "xelatex" "$XELATEX_CMD"

# For WSL + Windows pandoc.exe, pdf-engine must be a Windows-style executable name/path.
PDF_ENGINE_ARG="$XELATEX_CMD"
if [ -n "$PANDOC_CMD" ] && [ -n "$XELATEX_CMD" ] && is_wsl; then
  if [[ "$PANDOC_CMD" == *.exe ]]; then
    if [[ "$XELATEX_CMD" == /mnt/* ]] && command -v wslpath >/dev/null 2>&1; then
      PDF_ENGINE_ARG="$(wslpath -w "$XELATEX_CMD")"
    else
      PDF_ENGINE_ARG="xelatex"
    fi
  fi
fi

if [ -n "$QUARTO_CMD" ]; then
  echo "[build] rendering revealjs HTML..."
  "$QUARTO_CMD" render "$QMD_FILE" \
    --to revealjs \
    --output-dir "$OUT_DIR" \
    --output "kuairand_interview.html"

  echo "[build] rendering beamer PDF..."
  "$QUARTO_CMD" render "$QMD_FILE" \
    --to beamer \
    --output-dir "$OUT_DIR" \
    --output "kuairand_interview.pdf"

  echo "[done] HTML: $OUT_DIR/kuairand_interview.html"
  echo "[done] PDF : $OUT_DIR/kuairand_interview.pdf"
  exit 0
fi

echo ""
echo "[warn] 未检测到 quarto，进入 fallback 策略。"
echo "[warn] 目标格式（revealjs + beamer）需要 quarto 才能完整保证同源输出。"

if [ -n "$PANDOC_CMD" ]; then
  echo "[fallback] 尝试用 pandoc 直接导出 HTML..."
  "$PANDOC_CMD" -f markdown "$QMD_FILE" -s -o "$OUT_DIR/kuairand_interview.html" --toc \
    -c "assets/style.css" -V lang=zh || true
  echo "[fallback] HTML (pandoc) 输出: $OUT_DIR/kuairand_interview.html"
else
  echo "[error] pandoc 也不存在，无法 fallback 导出 HTML。"
fi

if [ -n "$PANDOC_CMD" ] && [ -n "$XELATEX_CMD" ]; then
  echo "[fallback] 尝试用 pandoc + xelatex 导出 PDF..."
  "$PANDOC_CMD" -f markdown "$QMD_FILE" -s -o "$OUT_DIR/kuairand_interview.pdf" \
    --pdf-engine="$PDF_ENGINE_ARG" -H "assets/latex-header.tex" || true
  echo "[fallback] PDF (pandoc) 输出: $OUT_DIR/kuairand_interview.pdf"
else
  echo "[error] 缺少 pandoc 或 xelatex，无法 fallback 导出 PDF。"
fi

echo ""
echo "[hint] 推荐安装 quarto 后重跑："
echo "       quarto render interview_deck.qmd --to revealjs --output-dir output --output kuairand_interview.html"
echo "       quarto render interview_deck.qmd --to beamer  --output-dir output --output kuairand_interview.pdf"
