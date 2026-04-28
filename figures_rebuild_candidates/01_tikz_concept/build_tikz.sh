#!/usr/bin/env bash
# 编译所有 TikZ 概念图为 PDF，编译完清理中间产物。
# 用法： bash build_tikz.sh
set -e
cd "$(dirname "$0")"

for tex in fig_rebuild_*.tex; do
    name="${tex%.tex}"
    echo "[xelatex] $name"
    xelatex -interaction=nonstopmode -halt-on-error "$tex" > /dev/null
    # 清理中间产物
    rm -f "${name}".aux "${name}".log "${name}".out "${name}".toc \
          "${name}".synctex.gz "${name}".fdb_latexmk "${name}".fls \
          "${name}".blg "${name}".bbl "${name}".bcf "${name}".run.xml \
          "${name}".xdv
done
echo "[done] 所有 .tex 已编译为 .pdf，中间产物已清理"
