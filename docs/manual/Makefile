# OpenBench 手册编译 Makefile
# 用法：
#   make probe   — 检查 LaTeX 工具链
#   make user    — 编译用户卷（user/main_user.pdf）
#   make dev     — 编译开发卷（developer/main_developer.pdf）
#   make ops     — 编译运维卷（operations/main_operations.pdf）
#   make master  — 编译总卷（manual.pdf）
#   make all     — 三卷 + 总卷全部编译
#   make clean   — 清理所有中间产物与 PDF

LATEXMK = latexmk
RM = rm -rf

.PHONY: all user dev ops master clean probe

all: user dev ops master

# 子目录编译时显式指定 ../latexmkrc，避免落入 latexmk 的 DVI 默认模式
SUBRC = -r ../latexmkrc

user:
	cd user && $(LATEXMK) $(SUBRC) main_user.tex

dev:
	cd developer && $(LATEXMK) $(SUBRC) main_developer.tex

ops:
	cd operations && $(LATEXMK) $(SUBRC) main_operations.tex

master:
	$(LATEXMK) manual.tex

clean:
	-cd user && $(LATEXMK) $(SUBRC) -C main_user.tex
	-cd developer && $(LATEXMK) $(SUBRC) -C main_developer.tex
	-cd operations && $(LATEXMK) $(SUBRC) -C main_operations.tex
	-$(LATEXMK) -C manual.tex
	$(RM) user/_minted-* developer/_minted-* operations/_minted-* _minted-*

probe:
	@echo "=== xelatex ==="
	@which xelatex && xelatex --version | head -1 || (echo "MISSING" && exit 1)
	@echo "=== latexmk ==="
	@which latexmk && latexmk --version | head -1 || (echo "MISSING" && exit 1)
	@echo "=== pygmentize ==="
	@which pygmentize && pygmentize -V | head -1 || (echo "MISSING" && exit 1)
	@echo "=== ctex / minted / tcolorbox / subfiles ==="
	@kpsewhich ctexbook.cls && kpsewhich minted.sty && kpsewhich tcolorbox.sty && kpsewhich subfiles.cls
