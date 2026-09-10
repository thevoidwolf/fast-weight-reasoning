# Makefile — pdflatex + bibtex build.
#
#   make          → build main.pdf
#   make clean    → remove build intermediates (keeps main.pdf)
#   make cleanall → also remove main.pdf
#   make watch    → rebuild on source change (requires latexmk)
#
# Requires a TeXLive install with the packages listed in
# arxiv-preprint.sty (all standard).

MAIN    := main
LATEX   := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX  := bibtex
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

TEXFILES := $(wildcard *.tex)
STYFILES := $(wildcard *.sty)
BIBFILES := $(wildcard *.bib)
FIGURES  := $(wildcard figures/*)

.PHONY: all clean cleanall watch pdf

all: $(MAIN).pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(TEXFILES) $(STYFILES) $(BIBFILES) $(FIGURES)
	$(LATEX) $(MAIN)
	-$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

watch:
	$(LATEXMK) -pvc $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz

cleanall: clean
	rm -f $(MAIN).pdf
